---
title: "Gradient Descent from First Principles: SGD, Momentum, and Adam"
date: 2025-05-01
permalink: /posts/2025/05/optimization-first-principles/
tags:
  - optimization
  - deep learning
  - machine learning
---

Every neural network training run is, at its core, a search problem: find the parameters θ that minimize a loss function L(θ). The tools we use to navigate that search — SGD, momentum, Adam — have unintuitive behaviors that matter enormously in practice. This post derives them from scratch.

## The Optimization Problem

We have a model with parameters θ ∈ ℝᵈ and a loss function L(θ) = (1/n) Σᵢ ℓ(f(xᵢ; θ), yᵢ). Our goal is to find θ* = argmin L(θ). For smooth, convex functions this is well-studied. Deep networks are neither, but the same tools apply with caveats.

A loss *landscape* is the surface traced by L(θ) over parameter space. In high dimensions, this is impossible to visualize, but useful intuitions carry over from 2D: valleys (local minima), plateaus (saddle points), cliffs (sharp gradients), and ravines (high curvature in one direction, low in another).

## Gradient Descent: The Simplest Idea

If L is differentiable, the gradient ∇L(θ) points in the direction of steepest ascent. Moving opposite to it decreases the loss:

```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

where η > 0 is the learning rate. This is the fundamental update rule — every optimizer is a variation on it.

**Why this works (locally):** By Taylor expansion, L(θ - η∇L) ≈ L(θ) - η||∇L||² for small η. As long as η is small enough and ∇L ≠ 0, we decrease the loss.

**The catch:** Computing ∇L(θ) requires a full pass over all n training examples. For n = 10⁷ and large models, this is computationally prohibitive.

## Stochastic Gradient Descent (SGD)

Instead of the full gradient, estimate it using a single randomly sampled example or a mini-batch of size B:

```
g_t = (1/B) Σ_{i ∈ batch} ∇ℓ(f(xᵢ; θ), yᵢ)
θ_{t+1} = θ_t - η · g_t
```

This is an unbiased estimator: 𝔼[g_t] = ∇L(θ). The noise from sampling is actually useful — it acts as implicit regularization and helps escape sharp local minima.

**The learning rate problem:** SGD is extremely sensitive to η. Too large: diverges. Too small: converges slowly. And the right η varies across parameters — some directions of the loss landscape are steep, others flat.

```python
def sgd_step(params, grads, lr=0.01):
    return [p - lr * g for p, g in zip(params, grads)]
```

## Momentum: Accelerating Along Ravines

In a ravine — high curvature in one dimension, low in another — vanilla SGD oscillates across the narrow direction while crawling along the long axis. Momentum dampens this by accumulating a velocity vector:

```
v_t = β · v_{t-1} + g_t
θ_{t+1} = θ_t - η · v_t
```

With β = 0.9 (typical), each update is a weighted sum of all past gradients, exponentially decaying. Consistent gradients (long axis of the ravine) accumulate; inconsistent ones (narrow axis) cancel.

**Nesterov momentum** improves this with a "look-ahead" correction — compute the gradient at the projected next position rather than the current one:

```
v_t = β · v_{t-1} + ∇L(θ_t - η·β·v_{t-1})
θ_{t+1} = θ_t - η · v_t
```

This corrects for overshooting before it happens, giving faster convergence on convex problems.

```python
def momentum_step(params, grads, velocity, lr=0.01, beta=0.9):
    new_velocity = [beta * v + g for v, g in zip(velocity, grads)]
    new_params = [p - lr * v for p, v in zip(params, new_velocity)]
    return new_params, new_velocity
```

## RMSProp: Per-Parameter Learning Rates

Momentum helps with ravines but doesn't solve the fundamental issue: different parameters need different step sizes. RMSProp maintains a running estimate of the squared gradient magnitude per parameter:

```
s_t = ρ · s_{t-1} + (1-ρ) · g_t²
θ_{t+1} = θ_t - (η / √(s_t + ε)) · g_t
```

Parameters with consistently large gradients get smaller effective learning rates. Parameters with small gradients get larger steps. This adapts to the local geometry.

## Adam: Combining Both Ideas

Adam (Adaptive Moment Estimation) combines momentum's first moment with RMSProp's second moment:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # first moment (momentum)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          # second moment (RMSProp)

m̂_t = m_t / (1 - β₁ᵗ)                          # bias correction
v̂_t = v_t / (1 - β₂ᵗ)                          # bias correction

θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

The bias corrections matter at early steps: m₀ = v₀ = 0, so without correction, early updates would be severely underestimated. Default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3) work well across a wide range of problems.

```python
import numpy as np

def adam_step(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    new_m = [b1 * mi + (1 - b1) * g for mi, g in zip(m, grads)]
    new_v = [b2 * vi + (1 - b2) * g**2 for vi, g in zip(v, grads)]
    m_hat = [mi / (1 - b1**t) for mi in new_m]
    v_hat = [vi / (1 - b2**t) for vi in new_v]
    new_params = [p - lr * mh / (np.sqrt(vh) + eps)
                  for p, mh, vh in zip(params, m_hat, v_hat)]
    return new_params, new_m, new_v
```

## Learning Rate Schedules

Even with Adam, the learning rate matters. A common pattern:

**Warmup + cosine decay:** Start with a small lr, ramp up over a warmup period, then decay following a cosine curve to near-zero. This prevents early instability and allows fine-grained convergence late in training.

```python
def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
```

## Practical Recommendations

- **Default to Adam** with lr=1e-3 for most deep learning. It's robust to hyperparameter choice.
- **SGD + momentum** often generalizes better for image classification (ResNets, ViTs) when tuned — Adam can converge to sharper minima.
- **Gradient clipping** (max norm 1.0) is essential for transformers and RNNs to prevent exploding gradients.
- **Warmup is not optional** for transformers — the Adam bias correction isn't enough at step 1.
- **Weight decay** (L2 regularization) should be applied to weights but not to biases or LayerNorm parameters. AdamW does this correctly; vanilla Adam does not.

The choice of optimizer is less important than tuning the learning rate and schedule. A well-tuned SGD often beats a poorly-tuned Adam, and vice versa.
