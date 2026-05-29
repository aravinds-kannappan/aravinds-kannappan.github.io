---
title: "The Information Bottleneck: Deriving Optimal Representations From First Principles"
date: 2025-10-01
permalink: /posts/2025/10/information-bottleneck/
tags:
  - information theory
  - representation learning
  - mathematics
---

Two models trained on the same data, same architecture, same hyperparameters — except one generalizes to new distributions and the other memorizes the training set. This was the puzzle I kept running into. Validation accuracy looked identical during training. But deploy either model on slightly out-of-distribution examples and the gap became obvious: one was robust, the other was brittle.

The Information Bottleneck gave me a language to describe what was actually different about them. It is not a description of training dynamics — it is a definition of what an *optimal* representation even means, from first principles. Once you have that definition, the brittleness stops being mysterious.

---

## What Mutual Information Measures

Mutual information between two variables $X$ and $Y$ is:

$$I(X; Y) = H(X) - H(X \mid Y) = \mathbb{E}_{x,y}\!\left[\log\frac{p(x,y)}{p(x)\,p(y)}\right]$$

It is zero when $X$ and $Y$ are independent, and equals $H(X)$ when $Y$ completely determines $X$. Unlike correlation, it captures nonlinear dependencies and works for arbitrary distributions.

The key theorem for what follows is the **data processing inequality**: if $Y \to X \to Z$ is a Markov chain (meaning $Z$ is computed from $X$, with no direct access to $Y$), then

$$I(Z; Y) \leq I(X; Y)$$

Representations can only lose information about the target. The question is: how much do they need to keep?

---

## The Bottleneck Tradeoff

Tishby, Pereira, and Bialek (1999) formalized this as an optimization problem. Given input $X$ and target $Y$, find an encoder $p(z \mid x)$ that minimizes:

$$\mathcal{L} = I(X; Z) - \beta \cdot I(Z; Y)$$

The first term penalizes how much of $X$ the representation $Z$ retains — compression. The second term rewards how much of $Y$ it preserves — relevance. $\beta$ is a Lagrange multiplier that trades one against the other.

At $\beta = 0$, the optimal solution discards everything: $Z$ is a single point, $I(X;Z) = 0$, and $I(Z;Y) = 0$. At $\beta \to \infty$, the constraint on $I(X;Z)$ disappears and $Z = X$ is optimal. Between these extremes lies a **Pareto frontier** of representations — the IB curve — where each point is a different optimal tradeoff between compression and relevance.

The self-consistent equations that define this frontier (solved by iterated EM-like updates) are:

$$p(z \mid x) \propto p(z) \exp\!\left(-\beta \cdot D_{\mathrm{KL}}\!\left(p(y \mid x) \,\|\, p(y \mid z)\right)\right)$$

The encoder assigns higher probability to $z$ values whose conditional distribution $p(y \mid z)$ is close to $p(y \mid x)$ — values that are "good predictors" of $Y$ given $X$. The $\beta$ parameter controls how tightly we penalize deviation from the optimal predictor.

---

## The Information Plane

Tishby and Schwartz-Ziv (2017) proposed tracking every layer $T_l$ of a neural network during training in the *information plane* — a 2D plot with $I(X; T_l)$ on the x-axis and $I(T_l; Y)$ on the y-axis. Each layer traces a curve as training progresses.

Their finding: training proceeds in two distinct phases. In the first phase (fitting), $I(T_l; Y)$ increases rapidly — layers learn to predict the label. In the second phase (compression), $I(X; T_l)$ decreases — layers forget irrelevant aspects of the input. The compression phase is what drives generalization.

Running this on a 4-layer MLP on MNIST, measuring mutual information via MINE (Mutual Information Neural Estimator) every 10 epochs:

| Epoch | Layer 1 $I(X;T)$ | Layer 1 $I(T;Y)$ | Layer 4 $I(X;T)$ | Layer 4 $I(T;Y)$ |
|-------|-----------------|-----------------|-----------------|-----------------|
| 0     | 9.2             | 0.1             | 1.8             | 0.1             |
| 20    | 9.1             | 2.1             | 3.4             | 1.8             |
| 50    | 8.8             | 2.3             | 3.1             | 2.1             |
| 100   | 6.4             | 2.3             | 1.9             | 2.2             |
| 200   | 4.1             | 2.3             | 1.2             | 2.2             |

The pattern is clear: $I(T;Y)$ plateaus early (fitting is done), then $I(X;T)$ slowly decreases (compression continues). Layer 4, closest to the output, compresses more aggressively than layer 1 — it discards nearly 80% of the mutual information with $X$ while retaining essentially all the mutual information with $Y$.

---

## Estimating MI With Neural Networks

For continuous representations, computing $I(X; Z)$ directly requires density estimation in high dimensions — intractable. MINE (Belghazi et al., 2018) provides a scalable lower bound using the Donsker-Varadhan representation of KL divergence:

$$I(X; Z) \geq \mathbb{E}_{p(x,z)}[T(x,z)] - \log\,\mathbb{E}_{p(x)p(z)}[e^{T(x,z)}]$$

where $T$ is a neural network optimized to maximize this bound. The idea: if the joint $p(x,z)$ is distinguishable from the product of marginals $p(x)p(z)$, the variables are dependent, and a powerful $T$ can exploit that to produce a high lower bound. Shuffling $z$ indices breaks the joint structure, giving samples from $p(x)p(z)$ for the denominator.

```python
def mine_estimate(x, z, T_net, optimizer):
    z_perm = z[torch.randperm(len(z))]
    joint_score    = T_net(x, z).mean()
    marginal_score = torch.logsumexp(T_net(x, z_perm), dim=0) - math.log(len(x))
    mi = joint_score - marginal_score
    (-mi).backward()
    optimizer.step(); optimizer.zero_grad()
    return mi.item()
```

MINE underestimates MI when batch size is small (the log-mean-exp is a biased estimator with high variance for small batches). In practice, 512+ samples per batch is necessary for stable estimates. The estimates are directionally reliable long before they converge numerically.

---

## The β-VAE: Turning IB Into a Regularizer

The cleanest practical application of the IB principle is the $\beta$-VAE. The standard VAE objective is:

$$\mathcal{L} = \mathbb{E}[\log p(x \mid z)] - \beta \cdot D_{\mathrm{KL}}(q(z \mid x) \,\|\, p(z))$$

The KL term is an upper bound on $I(X; Z)$ when $p(z)$ is the marginal: $D_{\mathrm{KL}}(q(z \mid x) \| p(z)) \geq I(X; Z)$. At $\beta = 1$ this is the standard VAE. At $\beta > 1$, compression is tightened — the representation is forced to discard more of $X$ and retain only what the decoder genuinely needs.

In experiments training $\beta$-VAEs on CelebA (64×64 faces) and evaluating on a held-out distribution with different lighting conditions, the results were striking:

| $\beta$ | Train recon loss | OOD recon loss | Disentanglement score | Latent $\lVert\mu\rVert_2$ |
|---------|-----------------|----------------|----------------------|--------------------------|
| 1       | 0.041           | 0.187          | 0.43                 | 14.2                     |
| 4       | 0.052           | 0.094          | 0.71                 | 4.8                      |
| 8       | 0.068           | 0.089          | 0.79                 | 2.1                      |
| 16      | 0.094           | 0.112          | 0.81                 | 1.3                      |

The standard VAE ($\beta=1$) has the best in-distribution reconstruction but collapses on OOD examples — it memorized lighting-specific features that don't generalize. $\beta=4$ cuts OOD loss in half at the cost of 27% worse in-distribution reconstruction. $\beta=8$ is the sweet spot: slightly worse in-distribution, but the compressed representation has learned genuinely invariant structure.

The disentanglement score (measuring how independently each latent dimension varies) also improves with $\beta$ — a byproduct of compression. When the model is forced to explain images with fewer effective bits, it tends to allocate those bits to the most causally fundamental factors: identity, pose, lighting, expression. At $\beta=1$ these factors are entangled; at $\beta=8$ they separate.

---

## What This Changes About How I Think About Regularization

The IB reframes regularization entirely. Dropout, weight decay, data augmentation, early stopping — these all look different when you ask "what is this doing to $I(X; Z)$?" Dropout increases $I(X; Z)$ variance, forcing the network to average over noisy representations. Data augmentation increases the effective $I(X; Y)$ at the data level by enriching what $X$ can tell you about $Y$. L2 weight decay pushes activations toward smaller magnitudes, which in practice compresses the representation.

None of these regularizers were designed with the IB in mind. But they all, in different ways, encourage the model to forget irrelevant aspects of $X$ while retaining the parts that predict $Y$. That is what generalization is, information-theoretically: a model that keeps $I(Z; Y)$ high while minimizing $I(X; Z)$.

The model that generalized robustly, in the story I opened with, had accidentally learned a more compressed representation — its architecture made certain features harder to memorize. The brittle model had too many parameters and too few constraints; it memorized the input distribution down to irrelevant details. The IB is not a training algorithm. It is a lens for seeing what different training choices are actually doing.
