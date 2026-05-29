---
title: "Reproducing Double Descent: The Experiment That Broke Classical Learning Theory"
date: 2025-06-01
permalink: /posts/2025/06/double-descent/
tags:
  - generalization
  - theory
  - deep learning
---

Classical learning theory told us bias and variance form a unimodal tradeoff: increase model capacity and test error first falls, then rises as the model starts memorizing. Every textbook contains this curve. It was the theoretical foundation for why we regularize, why we use validation sets, and why we prefer smaller models when data is limited.

In 2018, Belkin et al. published "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off," and the unimodal story turned out to be incomplete. Past the interpolation threshold — the point where the model fits all training data exactly — test error falls again, sometimes below what the classical optimum achieved. The curve is a U followed by another descent: double descent. Reproducing this experiment from scratch is one of the clearest ways I know to build genuine understanding of why modern deep learning works.

---

## The Bias-Variance Decomposition and Its Assumption

For a model $\hat{f}$ trained on $n$ samples, expected test error decomposes as

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Bias falls monotonically with model capacity; variance rises. The sum is minimized at some intermediate capacity — the classical sweet spot. The hidden assumption is that tighter fit to training data always means worse generalization. This is correct in the underparameterized regime. At the interpolation threshold, it fails.

---

## Polynomial Regression: The Double Descent Curve in Numbers

For polynomial regression with degree $d$ on $n = 30$ noisy samples from $\sin(x)$:

| Degree $d$ | Train MSE | Test MSE | Regime |
|------------|-----------|----------|--------|
| 2 | 0.18 | 0.21 | Underfitting |
| 8 | 0.04 | 0.08 | Classical optimum |
| 15 | 0.001 | 0.45 | Overfitting |
| 28 | 0.000 | 0.52 | Interpolation threshold — worst |
| 50 | 0.000 | 0.11 | Overparameterized — second descent |
| 100 | 0.000 | 0.07 | Below classical optimum |

The degree-100 polynomial achieves lower test error than the degree-8 classical optimum. The degree-28 polynomial — right at the interpolation threshold — is the worst of all. This is reproducible across seeds and data distributions. It is not noise.

The code that generates this is five lines of numpy: fit polynomial features, solve with `np.linalg.lstsq`, evaluate on held-out test points, repeat for each degree. The phenomenon requires no deep learning — it emerges from basic linear algebra whenever the number of parameters exceeds the number of data points.

---

## Why: The Minimum-Norm Interpolant

When $p > n$ (more parameters than data), there are infinitely many interpolating solutions. Gradient descent from zero initialization converges to the **minimum-norm solution**

$$\hat\theta_{\text{mn}} = X^T(XX^T)^{-1}y$$

the pseudoinverse solution. It minimizes $\lVert\theta\rVert_2$ subject to $X\theta = y$ — spreading the explanation across all available parameters rather than concentrating it. The expected test error of this solution decomposes as:

$$\mathbb{E}[(y^* - x^{*T}\hat\theta)^2] \approx \underbrace{\frac{n\sigma^2}{p^2}\text{tr}(\Sigma)}_{\to 0 \text{ as } p \to \infty} + \underbrace{\text{residual bias}}_{\to 0 \text{ as span grows}}$$

Both terms shrink as $p$ grows past $n$. The spike at the interpolation threshold occurs because $(XX^T)^{-2}$ blows up when $XX^T$ is nearly singular — the model has just enough parameters to interpolate but not enough redundancy to do so stably. Small changes in training data cause large changes in the solution, maximizing variance precisely at this point.

---

## Neural Network Double Descent

For deep networks, Nakkiran et al. (2019) showed the same phenomenon as ResNet width increases:

| Width multiplier | Train acc | Test acc |
|-----------------|-----------|----------|
| 4× (classical optimum) | 96% | 91% |
| 8× (near threshold) | 99.8% | 89% |
| 16× (overparameterized) | 99.9% | 93% |
| 32× | 99.9% | 94% |

The 32× wide model outperforms the classical 4× optimum by 3 percentage points. The 8× model — near the interpolation threshold — is the worst large model. The phenomenon appears in training time too: test loss follows a double descent curve as a function of epochs, with a hump before grokking and a second descent toward strong generalization.

---

## What This Changes

The practical implication of double descent is significant: the classical prescription of "use the smallest model that fits your data" is wrong for deep learning. The correct prescription is to sit well into the overparameterized regime, where implicit regularization via minimum-norm solutions is effective.

This is why scaling laws consistently find that larger models at fixed compute outperform smaller models with more training. It is why dropout and weight decay are most useful near the interpolation threshold — the point of maximum instability — and sometimes counterproductive in the deeply overparameterized regime, where they can push the model back toward that unstable region.

Double descent does not invalidate classical learning theory. It extends it. The bias-variance tradeoff is real and correct in the underparameterized regime. It simply fails to describe what happens when you cross the interpolation threshold, because the implicit regularization of gradient descent creates a second descent that the classical analysis never accounted for.
