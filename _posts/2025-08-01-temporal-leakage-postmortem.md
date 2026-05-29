---
title: "The Model That Learned From the Future: A Temporal Leakage Postmortem"
date: 2025-08-01
permalink: /posts/2025/08/temporal-leakage-postmortem/
tags:
  - production ML
  - data engineering
  - failure modes
---

The validation dashboard said 99.7% precision. We had trained a fraud detection model for a healthcare claims processor, and by every metric it was performing remarkably well. The product team was excited. We were cautious — 99.7% felt too good — but we couldn't find the flaw, so we deployed.

Production precision: 61%.

The gap between 99.7% and 61% is not a model failure. It is a data pipeline failure that the model faithfully reflected. The model learned to predict fraud correctly on validation data because the validation data contained features computed from information that, in production, would not yet exist at prediction time. The model had learned from the future.

---

## What Temporal Leakage Is

A temporal leakage occurs when a training feature is computed using data from *after* the prediction timestamp. In fraud detection, if the model's input at time $t$ includes a "rolling fraud rate" feature that is actually computed by looking forward — using claims filed after $t$ — then in production that feature will have a completely different value, or won't be computable at all.

Leakage produces artificially high validation accuracy because the leaked feature is genuinely informative about fraud. The model isn't wrong to use it — it correctly associates the feature with the label. The problem is that it cannot use it in production, because the future hasn't happened yet. The model learned a relationship that is real in the training data and impossible in deployment.

In our case, the SQL window function included a forward-looking clause:

```sql
-- LEAKY: 15 FOLLOWING includes future claims
AVG(is_fraud) OVER (
    PARTITION BY provider_id ORDER BY claim_date
    ROWS BETWEEN 14 PRECEDING AND 15 FOLLOWING
) AS rolling_fraud_rate
```

A claim on day $t$ got a fraud rate computed using claims from $t+1$ through $t+15$ — claims that reflect the same fraudulent pattern as the current claim, making the feature nearly perfectly predictive at training time and useless at deployment.

---

## Finding the Leak: Mutual Information Audit

We diagnosed the leak by measuring mutual information between each feature and the label separately on in-sample data and on a strict temporal holdout. Features with large in-sample MI but near-zero future-holdout MI are leaky.

| Feature | MI (train fold) | MI (temporal holdout) | Gap | Leaky? |
|---------|----------------|----------------------|-----|--------|
| rolling_fraud_rate | 0.43 | 0.02 | 0.41 | **Yes** |
| procedure_avg_cost_delta | 0.31 | 0.05 | 0.26 | **Yes** |
| provider_claim_volume | 0.21 | 0.19 | 0.02 | No |
| diagnosis_code_risk | 0.18 | 0.17 | 0.01 | No |
| days_since_last_claim | 0.15 | 0.14 | 0.01 | No |

Two features were badly leaky. Together they explained essentially all of the 99.7% validation precision — and their absence in production explained the 61% precision drop.

---

## The Fix: Point-in-Time Features

Every feature must be computed using only data observably available at prediction time. The corrected SQL uses a strictly backward-looking window:

```sql
-- CORRECT: 1 PRECEDING excludes current claim, 30 PRECEDING is historical only
AVG(is_fraud) OVER (
    PARTITION BY provider_id ORDER BY claim_date
    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
) AS rolling_fraud_rate_pit
```

For the training dataset construction, point-in-time (PIT) correctness extends to table joins — each claim must join to the version of provider features that existed at the time of the claim, not the latest version:

```python
# PIT-correct join: feature_valid_from <= claim_date < feature_valid_to
df = claims.merge(features_history, on="provider_id").query(
    "feature_valid_from <= claim_date < feature_valid_to"
)
```

This is the slowly-changing dimension type-2 pattern from data warehousing, applied to ML feature engineering.

---

## Results After the Fix

| Metric | Leaky model | PIT-correct model |
|--------|------------|-------------------|
| Validation precision | 99.7% | 84.3% |
| Production precision | 61.0% | 82.1% |
| Validation–production gap | 38.7 pp | 2.2 pp |

Validation precision dropped from 99.7% to 84.3%. This was expected and correct — we were no longer measuring how well the model predicts from the future, but how well it predicts the future from the past. Production precision rose to 82.1%, and the validation-production gap collapsed from 38.7 percentage points to 2.2. The model was finally trustworthy.

---

## What Automated Leakage Detection Looks Like

Leakage detection must be automatic and run before every deployment. The minimum viable check:

1. **Temporal MI audit**: for each feature, compute mutual information with the label on both a random split and a temporal holdout split. A large gap flags potential leakage.
2. **Timestamp provenance**: verify that the computation timestamp of each feature is strictly before the prediction timestamp. This requires instrumenting the feature store with provenance metadata.
3. **Distribution monitoring**: track each feature's distribution at prediction time versus training time. A leaky feature will shift in production in a predictable direction.

The 99.7% precision that fooled us was a number we wanted to be true. The honest number — 84.3% — was less exciting but represented something real. The single most important discipline in production ML is learning to distrust results that feel too good, building the checks that automatically flag them, and accepting the honest numbers even when they are not the ones you hoped for.
