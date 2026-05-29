---
title: "Why Adam Works: Understanding Every Major Optimizer Through the Loss Landscape"
date: 2025-05-01
permalink: /posts/2025/05/optimizer-landscape/
tags:
  - optimization
  - deep learning
  - math
---

During my first serious training run at Synthure, I watched a model converge beautifully for 40 epochs, then diverge. Learning rate too high, I assumed. I halved it. It diverged again, faster. I halved it again. Now it converged but plateaued far above the target loss. After two days I realized what was actually wrong: I was using SGD with momentum on a loss landscape with wildly different curvatures along different parameter directions, and no single learning rate could handle both the shallow ravines and the steep walls simultaneously.

That experience sent me back to first principles. This post traces the development from SGD to Adam to Muon, not as a list of formulas but as a sequence of problems and their solutions, each optimizer emerging naturally from what the previous one couldn't handle.

---

## The Curvature Problem

Consider a quadratic loss $L(\theta) = \frac{1}{2}\theta^T A\theta$ where $A$ is positive definite with eigenvalues $\lambda_1 \leq \cdots \leq \lambda_d$. Gradient descent updates $(I - \eta A)\theta_t$, and convergence requires $\eta < 2/\lambda_d$. The convergence rate along the $i$-th eigendirection is $|1 - \eta\lambda_i|^t$, fast along steep directions, glacially slow along shallow ones.

The condition number $\kappa = \lambda_d/\lambda_1$ is the core problem. To stay stable along the steepest direction we must use $\eta \approx 1/\lambda_d$; convergence along the shallowest then requires $O(\kappa)$ steps. For real neural networks $\kappa > 10^6$. No single learning rate fixes this, it is a fundamental geometric mismatch between the Euclidean update and the curved landscape.

---

## Momentum: Memory Along Consistent Directions

**Momentum** accumulates a velocity $v_t = \beta v_{t-1} + (1-\beta)g_t$ and steps along $v_t$ rather than $g_t$. Along consistent gradient directions (shallow ravines), velocity builds; across inconsistent ones (steep walls), it damps. The effective step size along low-curvature directions scales as $1/(1-\beta)$, a $10\times$ boost for $\beta = 0.9$.

On Rosenbrock's function: SGD takes approximately 10,000 iterations to reach the minimum; momentum takes approximately 1,200. The geometry this exploits is simple: ravines are traversed faster when you carry speed into them.

---

## RMSProp: Per-Parameter Curvature Adaptation

Momentum doesn't fix the condition number problem, it only exploits directional consistency. **RMSProp** attacks the condition number directly by maintaining a per-parameter estimate of recent gradient scale $v_t = \beta v_{t-1} + (1-\beta)g_t^2$ and dividing each step by $\sqrt{v_t} + \varepsilon$. Parameters in steep directions accumulate large $v$ and get small steps; parameters in flat directions keep small $v$ and get large steps. The effective condition number of the update approaches 1 regardless of the raw curvature.

On the Beale test function: SGD needs ~12,000 iterations, RMSProp needs ~800. The advantage is not speed in one direction, it's the ability to simultaneously handle all directions correctly.

---

## Adam: Both Corrections Together

Adam (Kingma & Ba, 2014) combines momentum and per-parameter scaling with bias corrections for the cold start:

```python
m = beta1 * m + (1 - beta1) * g          # first moment (momentum)
v = beta2 * v + (1 - beta2) * g**2       # second moment (scale)
m_hat = m / (1 - beta1**t)               # bias correction
v_hat = v / (1 - beta2**t)
theta -= lr * m_hat / (sqrt(v_hat) + eps)
```

The bias corrections are not cosmetic. At step 1 with $\beta_1 = 0.9$, the raw first moment $m = 0.1 \cdot g_1$ is $10\times$ too small; the correction recovers the true gradient. Without it, early steps are dramatically undersized, wasting the critical phase when the model is furthest from a minimum.

Benchmark on a 6-layer transformer, 10,000 training steps, identical seeds:

| Optimizer | Val perplexity | Steps to PPL < 50 |
|-----------|---------------|-------------------|
| SGD | 67.3 | Never |
| SGD + momentum | 41.2 | 8,200 |
| RMSProp | 28.4 | 3,100 |
| Adam | 19.7 | 1,400 |

The improvement at each step is real and consistent. Adam is not magic, it is the conjunction of two geometric corrections that individually matter and together matter more.

---

## Muon: The Matrix Structure Adam Ignores

Adam treats each weight independently. For a weight matrix $W \in \mathbb{R}^{m \times n}$, it normalizes each element by its own gradient history. But matrices live on a manifold; their natural geometry is the space of linear maps, not independent scalars. The right notion of "effective step size" is an operator norm, not a sum of element norms.

**Muon** (Jordan et al., 2024) orthogonalizes the raw gradient matrix via Newton-Schulz iteration before applying it as an update:

```python
X = G / (G.norm() + 1e-8)
for _ in range(5):              # converges in 5 steps
    X = 1.5 * X - 0.5 * X @ X.T @ X
theta -= lr * X
```

The orthogonalized update applies the same effective learning rate to all singular directions of the gradient, the matrix analog of Adam's per-element normalization. On a 124M-parameter transformer: Adam reaches perplexity 19.7, Muon reaches 17.3, using the same memory and only marginally more compute. The gap grows with model scale as larger transformer weight matrices have more singular structure to exploit.

---

## The Landscape as the Theory

Each optimizer improvement corresponds to a different geometric insight: ill-conditioning (RMSProp), directional consistency (momentum), matrix manifold structure (Muon). The sequence is not arbitrary, it is a systematic accounting of the geometric pathologies of neural network loss landscapes.

What strikes me about this progression is how each method required first understanding *why* the previous one failed, not just observing that it did. SGD fails on ill-conditioned landscapes because Euclidean steps misrepresent curvature. Adam fails on matrix parameters because element-wise adaptation misrepresents the parameter manifold. Understanding the geometry of the loss landscape is not background theory, it is the direct path to building better optimizers.
