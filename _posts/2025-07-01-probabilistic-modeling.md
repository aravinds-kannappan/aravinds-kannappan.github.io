---
title: "Probabilistic Modeling from First Principles: Bayes to Variational Inference"
date: 2025-07-01
permalink: /posts/2025/07/probabilistic-modeling/
tags:
  - probabilistic modeling
  - bayesian inference
  - machine learning
---

Most ML models produce point estimates: a class label, a predicted value, a next token. But the real world is uncertain, and models that don't represent uncertainty will express inappropriate confidence — calibrated wrongly, brittle to distribution shift, and unable to communicate what they don't know. This post builds the probabilistic ML toolkit from the ground up.

## Probability as a Language for Uncertainty

There are two interpretations of probability:

**Frequentist:** Probability is the long-run frequency of an event over repeated experiments. θ is a fixed unknown quantity — it doesn't have a distribution.

**Bayesian:** Probability represents degrees of belief. θ can have a distribution — it represents our uncertainty about the true value given evidence.

Both are useful. For inference over parameters (what does this dataset tell us about θ?), Bayesian reasoning is more natural. For decision theory and prediction, both frameworks converge.

## Bayes' Theorem

Given observed data X and model parameters θ:

```
P(θ | X) = P(X | θ) · P(θ) / P(X)
```

- **P(θ)** is the *prior* — our beliefs about θ before seeing data
- **P(X | θ)** is the *likelihood* — how probable is the data given θ?
- **P(X)** is the *marginal likelihood* (evidence) — normalizing constant
- **P(θ | X)** is the *posterior* — updated beliefs after seeing data

The posterior combines prior knowledge with data. As n → ∞, the likelihood dominates and the prior washes out. With little data, the prior matters a lot.

```python
# Bayesian coin flip
# Prior: Beta(alpha, beta) — encodes prior belief about p(heads)
# Likelihood: Binomial(n, p)
# Posterior: Beta(alpha + heads, beta + tails)  [conjugate prior]

def update_beta_prior(alpha_prior, beta_prior, heads, tails):
    return alpha_prior + heads, beta_prior + tails

alpha, beta = 1, 1  # uniform prior
heads, tails = 7, 3
alpha_post, beta_post = update_beta_prior(alpha, beta, heads, tails)
posterior_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean: {posterior_mean:.3f}")  # 0.727
```

## Maximum Likelihood Estimation (MLE)

If we want a point estimate rather than a full posterior, maximize the likelihood:

```
θ_MLE = argmax_θ P(X | θ) = argmax_θ Σᵢ log P(xᵢ | θ)
```

Taking the log turns a product into a sum — numerically stable and analytically convenient. MLE finds the parameter value that makes the observed data most probable. For a Gaussian with unknown mean: θ_MLE = (1/n) Σ xᵢ (the sample mean). For logistic regression: gradient ascent on the log-likelihood.

MLE has no prior — it's purely data-driven. With small n, this can overfit badly.

## Maximum A Posteriori (MAP)

Add a prior and take the mode of the posterior:

```
θ_MAP = argmax_θ P(θ | X) = argmax_θ [log P(X | θ) + log P(θ)]
```

The log prior acts as a regularizer. A Gaussian prior N(0, 1/λ) on θ corresponds to L2 regularization with strength λ. A Laplace prior corresponds to L1. MAP gives point estimates — it's a compromise between MLE and full Bayesian inference.

## Full Bayesian Inference: The Problem

The ideal is to maintain the full posterior P(θ | X) and use it for predictions:

```
P(y* | x*, X) = ∫ P(y* | x*, θ) P(θ | X) dθ
```

This integral marginalizes over all possible θ, weighted by their posterior probability. For most models, this integral is intractable.

## Markov Chain Monte Carlo (MCMC)

MCMC approximates the posterior by drawing samples. The key insight: you don't need to evaluate P(X) (the normalizing constant) — you only need P(X|θ)P(θ) for the Metropolis-Hastings acceptance ratio.

**Hamiltonian Monte Carlo (HMC)** uses gradient information to propose distant moves efficiently, making it much more efficient than random-walk Metropolis in high dimensions. Stan and NumPyro implement HMC/NUTS.

MCMC is exact (asymptotically) but slow. It doesn't scale to large datasets or high-dimensional latent spaces.

## Variational Inference

Instead of sampling, approximate P(θ|X) with a simpler distribution q(θ) from a tractable family. Minimize the KL divergence:

```
KL(q(θ) || P(θ|X)) = 𝔼_q[log q(θ)] - 𝔼_q[log P(θ|X)]
```

Expanding using Bayes' theorem, minimizing KL is equivalent to maximizing the **Evidence Lower Bound (ELBO)**:

```
ELBO(q) = 𝔼_q[log P(X|θ)] - KL(q(θ) || P(θ))
         = (reconstruction term) - (regularization term)
```

The reconstruction term rewards q that assigns high probability to θ values that explain the data. The KL term keeps q close to the prior — preventing it from collapsing to a point mass.

**Mean-field variational inference** assumes q(θ) = ∏ᵢ qᵢ(θᵢ) — fully factored. This is fast but ignores posterior correlations.

## Variational Autoencoders (VAEs)

VAEs apply variational inference to learn latent representations. For a dataset of observations x, we posit latent variables z and learn:

- **Encoder** q_φ(z|x): approximate posterior — maps x to a distribution over z
- **Decoder** p_θ(x|z): likelihood — maps z back to x

The ELBO becomes:

```
ELBO = 𝔼_{z~q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

The reparameterization trick enables backpropagation through the sampling step: z = μ_φ(x) + ε · σ_φ(x), where ε ~ N(0,I). Gradients flow through μ and σ but not ε.

```python
class VAEEncoder(nn.Module):
    def encode(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization trick

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    return recon_loss + kl_loss
```

## Calibration: Does Your Model Know What It Doesn't Know?

A model is **calibrated** if its confidence scores match empirical frequencies. A calibrated classifier predicting 70% confidence should be right 70% of the time on those examples.

**Expected Calibration Error (ECE):**
```
ECE = Σ_b (|Bₙ| / n) · |acc(Bₙ) - conf(Bₙ)|
```

Group predictions into confidence bins; ECE is the weighted average gap between accuracy and confidence.

**Reliability diagram:** Plot accuracy vs. confidence. A perfectly calibrated model falls on the diagonal. Deep networks are notoriously overconfident — their reliability curves bow above the diagonal.

**Temperature scaling** is the simplest fix: divide logits by a scalar T before softmax. T > 1 softens the distribution; T < 1 sharpens it. T is learned on a validation set.

```python
def calibrate_temperature(logits, labels, T_init=1.5):
    T = torch.nn.Parameter(torch.tensor([T_init]))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        loss = F.cross_entropy(logits / T, labels)
        optimizer.zero_grad()
        loss.backward()
        return loss
    optimizer.step(closure)
    return T.item()
```

## When to Use Which Tool

| Approach | When to use |
|----------|------------|
| MLE | Large data, no strong priors, need a point estimate |
| MAP | Small data or need regularization, point estimate acceptable |
| MCMC | Small-medium models, need exact posterior characterization, can afford compute |
| Variational Inference | Large models, need scalable approximate posteriors, willing to trade accuracy for speed |
| Calibration post-hoc | Large discriminative models where full Bayes is infeasible |

The probabilistic framework is the right way to think about uncertainty in ML — even when you ultimately use a point estimate, understanding the posterior tells you what you're assuming and what could go wrong.
