---
title: "Failure Modes in Production ML: Leakage, Distribution Shift, and Calibration"
date: 2025-08-01
permalink: /posts/2025/08/failure-modes/
tags:
  - production ML
  - data leakage
  - distribution shift
  - calibration
---

A model that passes all your offline evaluations can still fail silently in production. The three failure modes in this post — data leakage, distribution shift, and miscalibration — are responsible for the vast majority of ML systems that work on paper and break in deployment. I've encountered all three in production and this is what I've learned about detecting and fixing them.

## 1. Data Leakage

Data leakage occurs when information unavailable at prediction time contaminates training. It produces unrealistically high offline performance that evaporates in production.

### Types of Leakage

**Target leakage:** A feature correlates with the target but is actually caused by it (or would be unknown at inference time). Example: predicting hospital readmission using a feature like "discharge prescriptions filled at hospital pharmacy" — the prescription is written *because* of the patient's condition, not independent of it. The feature leaks the label.

**Train-test leakage:** Preprocessing is applied to the combined train+test split before splitting, so the test set "sees" training data statistics. Classic example: normalizing features using the full-dataset mean and std, then splitting. The test set has absorbed training set distribution information.

**Temporal leakage:** Using future information to predict the past. In time-series, a feature computed over a future window (e.g., rolling average including future timesteps) will appear highly predictive but is undefined at inference time.

### Detection

Suspiciously high offline metrics are the first signal. Then:

1. **Audit feature provenance:** For each feature, trace back to its source. Ask: would this value be available at the moment of prediction in production?
2. **Check feature importances:** If an unexpected feature dominates importance, investigate its construction.
3. **Ablation test:** Train without the suspicious feature. If performance drops dramatically, it was doing work it shouldn't.
4. **Temporal sanity check:** For time-series, ensure your split respects time. Never shuffle before splitting.

```python
# Correct temporal split — never shuffle time-series data before splitting
def temporal_train_test_split(df, date_col, split_date):
    train = df[df[date_col] < split_date]
    test = df[df[date_col] >= split_date]
    return train, test

# Wrong: shuffling destroys temporal integrity
# X_train, X_test = train_test_split(df, test_size=0.2)  # DON'T do this for time-series
```

### Fix

- Enforce strict temporal splits and feature cutoff times
- Fit preprocessing (scalers, imputers, encoders) *only* on training data; transform test data using training statistics
- Use sklearn Pipelines to make this automatic and auditable

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# The pipeline ensures the scaler is fit only on training data
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)  # scaler uses training mean/std only
```

---

## 2. Distribution Shift

Distribution shift occurs when the distribution at deployment differs from training. Your model learned a function that maps from training distribution inputs to outputs — if the input distribution changes, the learned mapping may no longer apply.

### Types of Shift

**Covariate shift:** P(X) changes but P(Y|X) stays the same. The conditional relationship between features and labels is still valid — you just see different inputs. Example: a fraud detector trained on 2023 transactions deployed in 2025 when card-not-present fraud patterns have evolved.

**Label shift (prior probability shift):** P(Y) changes but P(X|Y) stays the same. The class distribution shifts. Example: a diagnostic classifier trained on a hospital cohort with 10% positive rate deployed in a general population with 2% positive rate.

**Concept drift:** P(Y|X) changes — the actual relationship between features and labels changes. This is the hardest to detect and fix. Example: a recommendation model trained pre-pandemic where "travel content" is high-engagement; post-pandemic, the signal flips.

### Detection

```python
# Kolmogorov-Smirnov test for covariate shift in a numeric feature
from scipy.stats import ks_2samp

def detect_covariate_shift(train_feature, prod_feature, alpha=0.05):
    stat, p_value = ks_2samp(train_feature, prod_feature)
    shifted = p_value < alpha
    return {'statistic': stat, 'p_value': p_value, 'shifted': shifted}

# KL-divergence for distribution shift monitoring
def kl_divergence(p, q, eps=1e-10):
    p, q = np.array(p) + eps, np.array(q) + eps
    p, q = p / p.sum(), q / q.sum()
    return np.sum(p * np.log(p / q))
```

**In production:** Monitor input feature distributions with rolling windows. Alert when KL-divergence or PSI (Population Stability Index) exceeds a threshold. Track model confidence distributions — a shift in average confidence often precedes a shift in accuracy.

### Fixes

**Covariate shift:** Importance weighting — reweight training examples by P_prod(X) / P_train(X). This recovers unbiased gradients under the deployment distribution.

**Label shift:** Estimate P_prod(Y) from deployment outputs and reweight the prior in the model (Platt scaling or calibration on recent deployment data).

**Concept drift:** No static fix. Requires retraining on recent data or online learning. The key engineering task is building a pipeline that detects drift and triggers retraining automatically.

```python
# Population Stability Index — standard production drift monitor
def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    exp_freq = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_freq = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_freq = np.clip(exp_freq, 1e-4, None)
    act_freq = np.clip(act_freq, 1e-4, None)
    psi_val = np.sum((act_freq - exp_freq) * np.log(act_freq / exp_freq))
    return psi_val
    # PSI < 0.1: no significant shift
    # 0.1–0.2: moderate shift, investigate
    # > 0.2: significant shift, retrain
```

---

## 3. Calibration Failures

A calibrated model's confidence score P(Y=1|X=x) = 0.7 means: among all examples where the model predicts 70% confidence, approximately 70% should be positive. Miscalibrated models are overconfident or underconfident — their scores can't be trusted as probabilities.

### Why This Matters

In many applications, you don't just want a prediction — you want to act optimally given the uncertainty. Clinical decision support, fraud flagging, content moderation: all require meaningful confidence scores to make good threshold decisions. An overconfident model will trigger too many actions; an underconfident one too few.

### Measuring Calibration

**Reliability diagram:** Bin predictions by confidence, plot average accuracy vs. average confidence per bin. A calibrated model's curve lies on the diagonal.

**Expected Calibration Error (ECE):**
```
ECE = Σ_b (|Bₙ|/n) · |acc(Bₙ) - conf(Bₙ)|
```

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece
```

### Causes

- **Deep networks:** trained with cross-entropy loss on overparameterized models tend to become overconfident over long training runs
- **Class imbalance:** models push probabilities toward the majority class
- **Batch normalization** and **weight decay** affect calibration in non-obvious ways

### Fixes

**Temperature scaling** is the simplest and most effective post-hoc fix. Scale logits by T before softmax; learn T on a validation set:

```python
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

def find_temperature(logits, y_true):
    def nll(T):
        scaled_probs = softmax(logits / T)
        return log_loss(y_true, scaled_probs)
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x

T = find_temperature(val_logits, val_labels)
calibrated_probs = softmax(test_logits / T)
```

**Platt scaling:** Fit a logistic regression on top of raw model scores. More flexible than temperature scaling — learns an affine transformation rather than just a scale.

**Isotonic regression:** Non-parametric calibration; fits a monotone function mapping raw scores to calibrated probabilities. More powerful but requires more validation data and can overfit on small sets.

---

## Putting It Together: A Production Checklist

| Failure Mode | Detection | Fix |
|---|---|---|
| Target leakage | Feature provenance audit, ablation | Remove leaking features, enforce cutoff times |
| Train-test leakage | Pipeline audit | Fit preprocessors on train only (use Pipeline) |
| Temporal leakage | Verify split respects time order | Temporal splits, PSI monitoring |
| Covariate shift | KS test, PSI on features | Importance weighting, retrain on recent data |
| Label shift | Monitor output distribution | Calibration on recent data, prior adjustment |
| Concept drift | Monitor prediction accuracy over time | Online learning, triggered retraining |
| Miscalibration | ECE, reliability diagram | Temperature scaling, Platt scaling |

The common thread: **none of these failures are visible in offline evaluation without deliberately testing for them**. Build explicit checks for each into your evaluation pipeline, not as one-off scripts but as automated monitors that run against every model version before deployment.

Production ML is part engineering, part epidemiology — you're constantly watching for signals that something has quietly gone wrong.
