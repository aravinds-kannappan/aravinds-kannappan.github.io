---
title: "Information Theory for Machine Learning: Entropy, KL Divergence, and Why Our Loss Functions Work"
date: 2025-10-01
permalink: /posts/2025/10/information-theory-ml/
tags:
  - information theory
  - mathematics
  - machine learning
---

Why does cross-entropy loss work for classification? Why do VAEs minimize KL divergence? Why does mutual information appear in representation learning? These aren't arbitrary engineering choices — they're consequences of a coherent mathematical framework developed by Claude Shannon in 1948. Understanding information theory illuminates *why* the standard loss functions are correct, not just *that* they work.

## Shannon Entropy: Quantifying Uncertainty

The entropy of a discrete random variable X with distribution p is:

```
H(X) = -Σᵢ p(xᵢ) log p(xᵢ)
```

Shannon's axiomatic derivation shows that this is the *unique* measure of uncertainty satisfying three natural properties: it's continuous in p, it's maximized by the uniform distribution, and adding an impossible outcome doesn't change it. The unit depends on the log base: base 2 gives bits, base e gives nats.

**Intuition:** H(X) is the expected number of bits needed to encode a message drawn from p. A coin flip H(Bernoulli(0.5)) = 1 bit. A biased coin (p=0.9) has H ≈ 0.47 bits — predictable events need fewer bits to describe. A deterministic outcome has H = 0 — no information content.

```python
import numpy as np

def entropy(p, eps=1e-10):
    p = np.array(p)
    p = p / p.sum()  # normalize
    return -np.sum(p * np.log(p + eps))

# Examples
print(entropy([0.5, 0.5]))        # 0.693 nats ≈ 1 bit
print(entropy([0.9, 0.1]))        # 0.325 nats
print(entropy([1.0, 0.0]))        # 0.0 nats (deterministic)
```

## Cross-Entropy: The Cost of Using the Wrong Distribution

The cross-entropy between true distribution p and model distribution q is:

```
H(p, q) = -Σᵢ p(xᵢ) log q(xᵢ)
```

This is the expected number of bits needed to encode messages from p using a code optimized for q. If your model q is wrong, you pay extra bits. The relationship to entropy:

```
H(p, q) = H(p) + KL(p || q)
```

Since H(p) is fixed (it's the irreducible uncertainty in the true distribution), **minimizing cross-entropy is identical to minimizing KL divergence**. This is why cross-entropy is the right loss for classification: it's the unique loss that measures how far our model distribution is from the true data distribution, up to a constant.

In classification, p is one-hot (the true label) and q is the softmax output:

```
H(p, q) = -log q(y_true)   [for one-hot p]
```

This is just the negative log-likelihood of the correct class — familiar cross-entropy loss. The information-theoretic derivation makes clear that this isn't a heuristic choice; it's the principled measure of distribution mismatch.

## KL Divergence: Asymmetric Distance Between Distributions

The Kullback-Leibler divergence from p to q:

```
KL(p || q) = Σᵢ p(xᵢ) log(p(xᵢ) / q(xᵢ)) = 𝔼_p[log(p/q)]
```

KL divergence is not a distance — it's asymmetric: KL(p||q) ≠ KL(q||p) in general, and it doesn't satisfy the triangle inequality. But it has a crucial property: **KL(p||q) ≥ 0**, with equality iff p = q almost everywhere. This follows from Jensen's inequality applied to the convex function -log.

**Zero-avoiding vs. zero-forcing:** KL(p||q), minimized over q, is *zero-avoiding* — q will try to cover the full support of p, even if q spreads mass in regions p doesn't. KL(q||p), minimized over q, is *zero-forcing* — q concentrates on modes of p and may ignore other modes. This asymmetry matters enormously in variational inference: minimizing KL(q||p) (reverse KL) leads to mode-seeking behavior, while minimizing KL(p||q) (forward KL) leads to mean-seeking behavior.

```python
def kl_divergence(p, q, eps=1e-10):
    p, q = np.array(p) + eps, np.array(q) + eps
    p, q = p / p.sum(), q / q.sum()
    return np.sum(p * np.log(p / q))

# KL is not symmetric
p = [0.4, 0.4, 0.2]
q = [0.3, 0.3, 0.4]
print(f"KL(p||q) = {kl_divergence(p, q):.4f}")
print(f"KL(q||p) = {kl_divergence(q, p):.4f}")  # different
```

## Mutual Information: What Two Variables Share

The mutual information between X and Y measures how much knowing one reduces uncertainty about the other:

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = KL(P(X,Y) || P(X)P(Y))
```

The last form is illuminating: mutual information is the KL divergence between the joint distribution and the product of marginals. It equals zero iff X and Y are independent, and increases as they become more dependent. Unlike correlation, mutual information captures nonlinear dependencies.

**In representation learning:** InfoNCE (used in contrastive learning methods like SimCLR, CLIP) is a lower bound on mutual information between views of the same data. Maximizing InfoNCE pushes the learned representations to preserve maximum information about the original data, while training the model to distinguish paired from unpaired examples — a tractable proxy for an otherwise intractable objective.

```
I(X; Y) ≥ 𝔼[log(f(x,y) / 𝔼_y'[f(x,y')])]  (InfoNCE bound)
```

## The Evidence Lower Bound: VAEs Through an Information Lens

The VAE objective — maximize the ELBO:

```
ELBO = 𝔼_{z~q}[log p(x|z)] - KL(q(z|x) || p(z))
```

has a clean information-theoretic interpretation. Rewrite it:

```
ELBO = -H(x|z)_q - KL(q(z|x) || p(z))
     = I(x; z)_q + H(z)_q - H(z|x)_q - KL(q(z|x) || p(z))
```

The first term rewards high mutual information between x and z (the latent should contain information about the data). The second term penalizes q(z|x) for deviating from the prior p(z) (the latent should be organized according to the prior structure). Maximizing the ELBO is simultaneously maximizing an information measure and minimizing a divergence — these don't always agree, which is the source of the posterior collapse phenomenon in VAEs.

## Minimum Description Length: A Unified View

The Minimum Description Length (MDL) principle says: the best model is the one that gives the shortest description of the data. Formally, the total description length is:

```
L(model, data) = L(model) + L(data | model)
```

This is precisely the Bayesian negative log-posterior: -log p(θ) - log p(data|θ). Maximizing the posterior (MAP estimation) minimizes total description length. L2 regularization corresponds to a Gaussian prior — penalizing large weight vectors because they require more bits to describe. The information-theoretic and Bayesian perspectives are the same thing, viewed differently.

## Why This Matters in Practice

Information theory tells you what your loss function is *doing*, which helps you debug when it doesn't do what you want:

- **Label smoothing** reduces H(p, q) by replacing one-hot p with a slightly diffuse distribution — this prevents the model from becoming overconfident on training labels and improves calibration
- **Posterior collapse in VAEs** happens when the KL term dominates and the decoder ignores z entirely — the encoder collapses to the prior, and the model's rate (KL) goes to zero but the distortion (reconstruction) increases. The fix (β-VAE, KL annealing, free bits) all involve controlling the balance between rate and distortion
- **Contrastive loss temperature** controls the effective support of the distribution — low temperature sharpens the distribution, high temperature flattens it, directly affecting the information-theoretic properties of learned representations

The loss function is always implicitly encoding an assumption about the true distribution. Understanding information theory makes those assumptions explicit.
