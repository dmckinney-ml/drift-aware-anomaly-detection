Excellent. I’ll convert this into a **portfolio-grade README** — clean, confident, engineering-focused, and suitable for GitHub or Hugging Face.

---

# Alert-Budget Controlled Anomaly Detection

**Statistical vs ML-Based Methods Across Real-World Time Series**

## Overview

This project evaluates anomaly detection methods under operational constraints, focusing on:

- Extreme class imbalance
- Alert budget tradeoffs
- Seasonality handling
- Robust statistical baselines
- Machine learning ranking behavior

Rather than optimizing for raw F1, this study evaluates:

> **Minimum alert rate required to achieve target event recall**

Two NAB datasets are used to illustrate how anomaly separability impacts model performance:

1. `nyc_taxi` — high seasonality, subtle anomalies
2. `machine_temperature_system_failure` — structural regime shifts

---

## Problem Framing

In production systems:

- Anomalies are rare (<1%)
- False positives are costly
- Alert fatigue is real

The key question is not:

> “Does the model detect anomalies?”

But rather:

> “How many alerts must we generate to capture critical events?”

This project evaluates anomaly detection through an **alert-budget lens**.

---

## Datasets

### 1. NYC Taxi (NAB – realKnownCause/nyc_taxi.csv)

- Strong daily seasonality
- 30-minute resolution
- Test set size: 3,097 points
- Labeled anomalies in test: 3
- Test anomaly rate: ~0.097%

This dataset represents a **hard anomaly detection problem**.

---

### 2. Machine Temperature (NAB – machine_temperature_system_failure.csv)

- Industrial sensor signal
- Structural regime shift before failure
- Clearer anomaly separability
- Low anomaly prevalence

This dataset represents a **structurally separable anomaly case**.

---

## Baseline: Seasonal-Differenced Rolling MAD

To establish a robust statistical baseline:

1. Seasonal differencing (period = 24):

[
seasonal_diff[t] = x[t] - x[t-24]
]

2. Rolling Median Absolute Deviation (MAD):

[
robust_z(t) = \frac{|x_t - median_t|}{1.4826 \cdot MAD_t}
]

3. Percentile-based thresholding from training distribution

This baseline is:

- Interpretable
- Seasonality-aware
- Alert-budget controllable
- Resistant to heteroskedastic noise

---

## Model: Isolation Forest

Isolation Forest was evaluated with:

- Feature engineering (lags, deltas, rolling statistics)
- Standardization
- Alert-rate-controlled percentile thresholding
- Event-based evaluation

Hyperparameters were tuned conservatively to avoid test leakage.

---

## Evaluation Strategy

Instead of raw F1, we measure:

- Event recall
- Alert rate
- Average Precision (AP)
- Minimum alert rate required for full event recall

Evaluation includes:

- Point-wise scoring
- Window-based near-hit scoring (±60 minutes)

---

# Results

## Case 1 — NYC Taxi

| Model            | Recall | Alerts   | Alert Rate |
| ---------------- | ------ | -------- | ---------- |
| Seasonal MAD     | 0.33   | 179      | 5.8%       |
| Isolation Forest | ~0.33  | ~112–200 | ~3–6%      |

### Key Observations

- Seasonality dominated naïve deviation methods.
- Robust scaling improved stability but not separability.
- True anomalies were not consistently ranked near the score tail.
- ML provided marginal improvement over robust statistics.

**Conclusion:**
This dataset exhibits a statistical separability ceiling. Model complexity cannot compensate for weak anomaly signal.

---

## Case 2 — Machine Temperature

| Model            | Recall | Alerts | Alert Rate |
| ---------------- | ------ | ------ | ---------- |
| Seasonal MAD     | 1.0    | 1361   | ~20–25%    |
| Isolation Forest | 1.0    | ~460   | ~7%        |

### Key Observations

- Structural regime shifts were highly separable.
- Isolation Forest reduced alerts by ~3× at equal event recall.
- Feature representation had greater impact than hyperparameter tuning.
- ML meaningfully improved operational efficiency.

**Conclusion:**
When anomalies are structurally separable, non-linear models significantly improve alert efficiency.

---

# Lessons Learned

## 1. Seasonality Must Be Addressed Explicitly

Strong periodic structure masks anomalies under naïve deviation metrics.

Seasonal differencing was required for meaningful baseline performance.

---

## 2. Robust Scaling Stabilizes but Does Not Guarantee Detection

MAD reduced volatility but did not improve ranking quality where anomalies were subtle.

Separability is a property of the data, not just the metric.

---

## 3. Alert Rate Is the Primary Design Constraint

Under extreme imbalance:

- Precision alone is insufficient
- Recall must be evaluated relative to alert volume

Model value is best measured as:

> Minimum alert rate required to achieve desired event recall

---

## 4. Ranking Quality Drives Operational Value

Average Precision exposed:

- Weak tail separation in taxi
- Strong tail concentration in machine temperature

When true anomalies are not near the extreme tail, threshold tuning cannot fix ranking limitations.

---

## 5. Model Value Is Dataset-Dependent

Machine learning is not universally superior to statistical baselines.

It adds value when:

- Anomaly structure is non-linear
- Structural regime shifts exist
- Feature representation enables separation

It adds little when:

- Anomalies resemble natural variance
- Seasonal structure dominates signal behavior

---

# Engineering Takeaways

This project demonstrates:

- Alert-budget-driven evaluation
- Seasonality-aware baseline modeling
- Robust statistical normalization
- ML vs statistical comparison under equal constraints
- Ranking diagnostics beyond threshold metrics
- Dataset-dependent anomaly separability analysis

This framework can be extended to:

- Custom time series via CSV upload
- Real-time alerting systems
- Drift monitoring pipelines
- Production anomaly services

---

# Future Work

- Add window-based NAB scoring
- Evaluate autoencoder-based detectors
- Explore ensemble anomaly scoring
- Add drift simulation and adaptive thresholding
- Package as interactive Hugging Face Space

---

# Why This Matters

Anomaly detection is often evaluated with headline metrics.

This project reframes it as:

> A constrained optimization problem under operational alert budgets.

By comparing statistical baselines and ML methods across datasets with different separability properties, it highlights when model complexity is justified — and when it is not.

---

If you’d like, I can next:

- Write a short Hugging Face Space description
- Add a polished repo structure outline
- Draft a “How to Use This Repo” section
- Or help you frame this for ML Engineer interviews
