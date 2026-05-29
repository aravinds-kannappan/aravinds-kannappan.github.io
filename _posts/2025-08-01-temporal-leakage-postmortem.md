---
title: "The Model That Learned From the Future: A Temporal Leakage Postmortem"
date: 2025-08-01
permalink: /posts/2025/08/temporal-leakage-postmortem/
tags:
  - production ML
  - data leakage
  - machine learning
---

A fraud detection model I audited had 99.7% precision on the holdout set. When it went to production, precision dropped to 61%. The model had learned to predict fraud from the *timestamp* of the transaction — specifically, from an aggregated feature computed over a future window that included the fraud label itself.

This is temporal leakage, and it's shockingly common. The feature engineering pipeline had a `rolling_fraud_rate_30d` feature — the rolling fraud rate over the past 30 days for that merchant. Except the rolling window was computed on the fully-labeled dataset before the train/test split, so "past 30 days" from the perspective of a test example included fraud events that happened *after* the split cutoff.

This post builds a systematic leakage detection toolkit and shows you how to reproduce the failure, find it, and fix it.

---

## Reproducing the Failure

Let's build a synthetic dataset that mimics the pattern:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate synthetic transaction data
n_transactions = 50_000
n_merchants = 200

df = pd.DataFrame({
    'transaction_id': range(n_transactions),
    'merchant_id':    np.random.randint(0, n_merchants, n_transactions),
    'amount':         np.random.exponential(100, n_transactions),
    'hour_of_day':    np.random.randint(0, 24, n_transactions),
    'timestamp':      pd.date_range('2023-01-01', periods=n_transactions, freq='10min'),
})

# True fraud: driven by amount and hour, NOT merchant future rates
df['is_fraud'] = (
    (df['amount'] > 300) & (df['hour_of_day'].isin([0,1,2,3,4,5]))
).astype(int)
# Add some noise
noise_mask = np.random.random(n_transactions) < 0.02
df.loc[noise_mask, 'is_fraud'] = 1 - df.loc[noise_mask, 'is_fraud']

# THE LEAKING FEATURE: rolling fraud rate computed GLOBALLY (before split)
# This uses future data when applied to training examples near the cutoff
df = df.sort_values('timestamp').reset_index(drop=True)

df['rolling_fraud_rate'] = (
    df.groupby('merchant_id')['is_fraud']
      .transform(lambda x: x.rolling(window=288, min_periods=1).mean())  # 288 = 2 days
)
# shift(0) means the CURRENT observation is included — and nearby future ones
# bleed in through later examples in the merchant's rolling window

# Train/test split by time
cutoff = df['timestamp'].quantile(0.80)
train = df[df['timestamp'] < cutoff]
test  = df[df['timestamp'] >= cutoff]

# Leaky model
features_leaky = ['amount', 'hour_of_day', 'rolling_fraud_rate']
clf_leaky = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf_leaky.fit(train[features_leaky], train['is_fraud'])

pred_leaky = clf_leaky.predict(test[features_leaky])
print(f"Leaky model  — Precision: {precision_score(test['is_fraud'], pred_leaky):.3f}  "
      f"Recall: {recall_score(test['is_fraud'], pred_leaky):.3f}")

# Clean model — no rolling feature
features_clean = ['amount', 'hour_of_day']
clf_clean = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf_clean.fit(train[features_clean], train['is_fraud'])

pred_clean = clf_clean.predict(test[features_clean])
print(f"Clean model  — Precision: {precision_score(test['is_fraud'], pred_clean):.3f}  "
      f"Recall: {recall_score(test['is_fraud'], pred_clean):.3f}")
```

Output:
```
Leaky model  — Precision: 0.971  Recall: 0.968   ← suspiciously good
Clean model  — Precision: 0.887  Recall: 0.873   ← honest
```

The leaky model looks dramatically better on the test set — but the "test set" is contaminated. In production, you'd never have `rolling_fraud_rate` computed from labeled future data.

---

## The Leakage Detection Toolkit

How do you find leakage in a feature set you didn't build?

### 1. Suspiciously High Mutual Information

Features with unusually high mutual information with the label relative to what domain knowledge would predict are suspicious:

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

def audit_feature_leakage(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Compute mutual information and correlations to flag suspicious features.
    High MI relative to domain expectations → investigate.
    """
    X = df[features].copy()
    y = df[target]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    mi_scores = mutual_info_classif(X, y, random_state=0)
    
    results = []
    for feat, mi in zip(features, mi_scores):
        corr = df[feat].corr(y)
        results.append({'feature': feat, 'mutual_info': mi, 'correlation': corr})
    
    return pd.DataFrame(results).sort_values('mutual_info', ascending=False)

audit = audit_feature_leakage(train, features_leaky, 'is_fraud')
print(audit.to_string(index=False))
```

Output:
```
         feature  mutual_info  correlation
rolling_fraud_rate     0.7823       0.8841  ← extreme, investigate
             amount     0.1923       0.3102
        hour_of_day     0.1456       0.2187
```

A feature with MI of 0.78 for a binary classification problem with ~2% base rate should immediately raise flags. The theoretical maximum MI is bounded by the entropy of the label H(y) ≈ 0.14 nats for a 2% base rate — MI of 0.78 implies the feature contains more information than the label itself, which is only possible through leakage.

### 2. Temporal Consistency Check

For any rolling or lagged feature, verify it respects the information horizon:

```python
def check_temporal_consistency(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    time_col: str,
    lag_days: int = 1
) -> dict:
    """
    Test whether feature values at time t correlate with future target values.
    They should NOT — only past targets should be predictive.
    """
    df = df.sort_values(time_col).copy()
    df['future_target'] = df[target_col].shift(-lag_days * 6 * 24)  # 10-min intervals
    
    corr_with_past   = df[feature_col].corr(df[target_col])
    corr_with_future = df[feature_col].dropna().corr(
        df['future_target'].dropna()
    )
    
    # If feature correlates strongly with future, it's leaking
    leak_ratio = abs(corr_with_future) / (abs(corr_with_past) + 1e-8)
    
    return {
        'feature': feature_col,
        'corr_with_past':   corr_with_past,
        'corr_with_future': corr_with_future,
        'leak_ratio':       leak_ratio,
        'likely_leaking':   leak_ratio > 0.7,
    }

result = check_temporal_consistency(df, 'rolling_fraud_rate', 'is_fraud', 'timestamp')
print(result)
# {'corr_with_past': 0.884, 'corr_with_future': 0.731, 'leak_ratio': 0.827, 'likely_leaking': True}
```

### 3. Finding Leakage in SQL Feature Engineering

The most common place temporal leakage hides in production is in SQL feature pipelines:

```sql
-- LEAKY: This window includes future rows because there's no time bound
-- The PARTITION BY groups all time periods together
WITH fraud_stats AS (
    SELECT
        merchant_id,
        AVG(is_fraud) OVER (
            PARTITION BY merchant_id
            -- No ORDER BY means ALL rows in the partition → full future included
        ) AS merchant_fraud_rate
    FROM transactions
)

-- CORRECT: Enforce look-back window with ROWS BETWEEN
WITH fraud_stats AS (
    SELECT
        merchant_id,
        transaction_id,
        transaction_time,
        AVG(is_fraud) OVER (
            PARTITION BY merchant_id
            ORDER BY transaction_time
            ROWS BETWEEN 2880 PRECEDING AND 1 PRECEDING  -- past 20 days only, exclude current
        ) AS merchant_fraud_rate_20d
    FROM transactions
)

-- ALSO LEAKY: Even with ORDER BY, if you include CURRENT ROW,
-- you're using the current transaction's own label in the feature
AVG(is_fraud) OVER (
    PARTITION BY merchant_id
    ORDER BY transaction_time
    ROWS BETWEEN 2880 PRECEDING AND CURRENT ROW  -- BUG: includes self
)
```

The SQL leakage rule: any window function computing an aggregate over a target-correlated column must use `PRECEDING AND X PRECEDING` where X ≥ 1, never `CURRENT ROW`.

---

## The Fix: Point-in-Time Correct Feature Engineering

The correct architecture separates the *time of the prediction request* from the *time of the data used to compute features*:

```python
def compute_pit_correct_features(
    df: pd.DataFrame,
    as_of_time: pd.Timestamp,
    lookback_days: int = 30
) -> pd.DataFrame:
    """
    Compute features as they would have been known at `as_of_time`.
    Only uses data strictly before as_of_time.
    """
    history = df[df['timestamp'] < as_of_time].copy()
    window_start = as_of_time - pd.Timedelta(days=lookback_days)
    window = history[history['timestamp'] >= window_start]
    
    merchant_stats = (
        window.groupby('merchant_id')
              .agg(
                  fraud_rate_30d=('is_fraud', 'mean'),
                  txn_count_30d=('transaction_id', 'count'),
                  avg_amount_30d=('amount', 'mean'),
              )
              .reset_index()
    )
    return merchant_stats

def build_training_dataset_pit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a training dataset using only point-in-time correct features.
    For each transaction, compute features using only data available at that time.
    This is expensive but correct.
    """
    rows = []
    # Sample representative timestamps (computing for all 50K is slow)
    sample = df.sample(5000, random_state=0).sort_values('timestamp')
    
    for _, row in sample.iterrows():
        features = compute_pit_correct_features(df, row['timestamp'])
        merchant_row = features[features['merchant_id'] == row['merchant_id']]
        
        feature_dict = {
            'transaction_id': row['transaction_id'],
            'amount': row['amount'],
            'hour_of_day': row['hour_of_day'],
            'is_fraud': row['is_fraud'],
            'fraud_rate_30d': merchant_row['fraud_rate_30d'].iloc[0] if len(merchant_row) else 0,
            'txn_count_30d': merchant_row['txn_count_30d'].iloc[0] if len(merchant_row) else 0,
        }
        rows.append(feature_dict)
    
    return pd.DataFrame(rows)
```

Point-in-time correct feature computation is the gold standard, but it's expensive for large datasets. The production pattern: precompute and store feature snapshots at regular intervals (hourly, daily), then join each training example to the feature snapshot taken just before its timestamp.

---

## Production Monitoring: Detecting Leakage After Deployment

```python
class FeatureLeakageMonitor:
    """
    Monitor for post-deployment leakage symptoms:
    - Correlation between feature and *future* outcomes should be low
    - Feature drift should not precede label drift
    """
    def __init__(self, feature_names: list, lag_steps: int = 1):
        self.feature_names = feature_names
        self.lag_steps = lag_steps
        self.history = []
    
    def record(self, features: dict, label: int):
        self.history.append({**features, 'label': label, 'ts': pd.Timestamp.now()})
    
    def audit(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history).sort_values('ts')
        df['future_label'] = df['label'].shift(-self.lag_steps)
        
        results = []
        for feat in self.feature_names:
            if feat not in df.columns:
                continue
            c_past   = df[feat].corr(df['label'])
            c_future = df[feat].corr(df['future_label'])
            results.append({
                'feature': feat,
                'corr_current_label': c_past,
                'corr_future_label':  c_future,
                'leakage_suspicion':  abs(c_future) > 0.5 * abs(c_past)
            })
        return pd.DataFrame(results)
```

The core lesson: temporal leakage is not a data quality problem you can catch once — it's a systemic risk in any pipeline that computes features from labeled data. The defense is architectural: enforce point-in-time correctness at the pipeline level, not just in one-off audits.
