---
title: "Why Models Fail on New Data: Generalization from First Principles"
date: 2025-06-01
permalink: /posts/2025/06/generalization/
tags:
  - generalization
  - machine learning
  - regularization
---

A model that fits training data perfectly is useless if it fails on anything it hasn't seen. Generalization — the ability to perform well on unseen examples drawn from the same distribution — is what we actually care about. Understanding why it happens (or doesn't) requires looking at the problem from first principles.

## What Generalization Means, Formally

We have a true data distribution 𝒟 over (x, y) pairs. We observe n training examples Sₙ = {(x₁,y₁),...,(xₙ,yₙ)} drawn i.i.d. from 𝒟. We want a hypothesis h that minimizes the *true risk*:

```
R(h) = 𝔼_{(x,y)~𝒟}[ℓ(h(x), y)]
```

But we can only measure the *empirical risk* on Sₙ:

```
R̂(h) = (1/n) Σᵢ ℓ(h(xᵢ), yᵢ)
```

The *generalization gap* is R(h) - R̂(h). A model with R̂(h) ≈ 0 but large R(h) has memorized training data — it has overfit.

## The Bias-Variance Decomposition

For squared loss, the expected test error decomposes into three terms:

```
𝔼[(y - ĥ(x))²] = Bias²[ĥ(x)] + Var[ĥ(x)] + σ²
```

where:
- **Bias** = 𝔼[ĥ(x)] - h*(x): how far off your average prediction is from the truth
- **Variance** = 𝔼[(ĥ(x) - 𝔼[ĥ(x)])²]: how much your prediction varies across different training sets
- **σ²**: irreducible noise in the data

**High bias (underfitting):** Your model class is too restricted to capture the true pattern. A linear model on nonlinear data will have high bias regardless of how much data you have.

**High variance (overfitting):** Your model is sensitive to which specific training examples you happened to draw. A degree-100 polynomial on 20 data points is wildly different depending on which 20 points you use.

The classical story is a tradeoff: increasing model complexity reduces bias but increases variance. This is true — but modern deep learning has complicated it.

```python
# Bias-variance estimation via bootstrap
def bias_variance(model_fn, X_train, y_train, X_test, y_test, n_bootstrap=200):
    predictions = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        model = model_fn()
        model.fit(X_train[idx], y_train[idx])
        predictions.append(model.predict(X_test))
    predictions = np.array(predictions)
    bias_sq = (predictions.mean(axis=0) - y_test)**2
    variance = predictions.var(axis=0)
    return bias_sq.mean(), variance.mean()
```

## VC Dimension: A Formal Measure of Capacity

How do we know when a model class is "too complex"? The VC dimension (Vapnik-Chervonenkis dimension) formalizes this.

A hypothesis class ℋ *shatters* a set of m points if for every possible binary labeling of those points, some h ∈ ℋ achieves zero training error. The VC dimension VC(ℋ) is the largest m for which ℋ can shatter some set of m points.

The key theorem: with high probability (1-δ),

```
R(h) ≤ R̂(h) + O(√(VC(ℋ)/n · log(n/VC(ℋ)) + log(1/δ)/n))
```

This bounds the generalization gap. As n → ∞, the gap vanishes. As VC(ℋ) grows, you need more data to close the gap. Linear classifiers in d dimensions have VC dimension d+1. Deep networks have enormous VC dimension — yet they generalize. This is the central mystery of modern deep learning.

## Regularization: Constraining the Hypothesis Class

If unconstrained optimization leads to overfitting, constrain it. Common approaches:

**L2 regularization (weight decay):** Add λ||w||² to the loss. The optimal weights satisfy a balance between fitting data and keeping weights small. Geometrically, this shrinks weights toward zero — high-magnitude weights need strong evidence to survive.

```python
loss = cross_entropy(predictions, targets) + lambda_ * sum(w**2 for w in weights)
```

**L1 regularization (LASSO):** Add λ||w||₁. Unlike L2, this induces exact sparsity — some weights become exactly zero. Useful when you believe few features are relevant.

**Dropout:** During training, randomly zero activations with probability p. Forces the network to learn redundant representations — no single neuron can be relied upon. At test time, multiply all weights by (1-p). Dropout is implicit ensembling: each forward pass trains a different sub-network.

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
    return x * mask
```

**Early stopping:** Monitor validation loss during training; stop when it starts increasing. Effectively limits the number of gradient steps, constraining the optimization trajectory.

## Cross-Validation: Estimating the Generalization Gap

Before evaluating on a held-out test set (which you use exactly once), use k-fold cross-validation to estimate generalization performance during model development:

1. Partition training data into k folds
2. For each fold k: train on the other k-1 folds, evaluate on fold k
3. Average the k evaluation scores

This gives an estimate of R(h) that uses all available data. k=5 or k=10 are standard. For small datasets, leave-one-out (k=n) gives the least-biased estimate but is computationally expensive.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV estimate: {-scores.mean():.3f} ± {scores.std():.3f}")
```

## The Double Descent Phenomenon

Classical wisdom says: as model complexity increases past the interpolation threshold (where train error → 0), test error rises catastrophically. This is the bias-variance tradeoff.

Modern deep learning breaks this. With enough overparameterization, test error *decreases again* after the interpolation threshold — a "double descent" curve. Heavily overparameterized models find interpolating solutions with good structure (smooth, minimum-norm), which happen to generalize.

This suggests the bias-variance tradeoff isn't wrong — it applies to a fixed model class as you vary the number of training samples. But comparing *different* model classes (a 10-parameter vs. 10-million-parameter model) at a fixed dataset size tells a different story.

## Practical Guidelines

1. **Always have a validation set** separate from your test set. Tune on validation, report on test — once.
2. **Regularize by default.** Weight decay (1e-4 to 1e-2) is free to try and rarely hurts.
3. **More data beats better algorithms.** If you're overfitting, get more data before engineering a better regularizer.
4. **Watch the train/val loss curves.** Diverging curves = overfitting. Both high = underfitting. Both low = done.
5. **Don't tune on the test set.** Ever. Even "just to check."

The generalization problem doesn't have a clean solution — it's an empirical science. Understanding the theory tells you what levers exist; experience tells you which ones to pull.
