Perfect — now we elevate this from a single-dataset analysis to a comparative anomaly detection study.

I’ll integrate Machine Temperature lessons directly into the section so it reads like a cohesive engineering narrative.

---

# Lessons Learned (Across Datasets)

## 1. Seasonality Can Dominate Deviation-Based Methods

In `nyc_taxi`, daily seasonality was the dominant signal component.

Without seasonal differencing:

- Rolling statistics tracked cyclic peaks rather than structural disruptions.
- Threshold sensitivity was extreme.
- Alert rates inflated rapidly.

Seasonal differencing was necessary to produce a stable deviation baseline.

By contrast, the **machine temperature dataset** exhibited weaker periodic structure and clearer structural breaks. Seasonality handling was less critical, and anomaly separability was inherently stronger.

**Insight:**
Anomaly detectability is heavily dependent on underlying signal structure. Strong periodicity can mask anomalies in purely statistical detectors.

---

## 2. Robust Scaling Improves Stability — Not Separability

Replacing rolling standard deviation with rolling MAD:

- Reduced sensitivity to local variance shifts.
- Improved threshold stability.
- Lowered false positive volatility.

However:

- In `nyc_taxi`, only 1 of 3 anomalies ranked highly under robust scaling.
- Average Precision remained very low (~0.002–0.003 range).

In contrast, on machine temperature data:

- True anomaly windows ranked near the extreme tail.
- Event recall reached 1.0 under reasonable alert budgets.

**Insight:**
Robust normalization controls noise but cannot create separability where the anomaly signal is weak relative to the background process.

---

## 3. Alert Budget Is the True Optimization Target

Under extreme class imbalance (~0.1% anomaly rate):

- Precision alone is misleading.
- Recall must be evaluated relative to alert volume.

For `nyc_taxi`:

- ~6% alert rate yielded only ~33% recall.
- Increasing recall required rapidly escalating alert rates.

For machine temperature:

- Rolling MAD baseline required ~1361 alerts to achieve 100% event recall.
- Isolation Forest reduced this to ~460 alerts (≈3× reduction).
- Recall remained 1.0 at significantly lower alert rates.

**Insight:**
Operational anomaly detection is an alert allocation problem under severe imbalance.

Model value is measured by:

> Minimum alert rate required to achieve desired event recall.

---

## 4. Ranking Quality Determines Practical Performance

Average Precision (AP) revealed dataset-dependent ranking behavior:

- `nyc_taxi`: True anomalies were not consistently near the top of the score distribution.
- Machine temperature: Anomalies ranked in the extreme tail under Isolation Forest.

This explains why:

- ML provided marginal gains in taxi.
- ML provided substantial alert reduction in machine temperature.

**Insight:**
When anomaly ranking is inherently weak, model choice matters less than representation.

When anomalies are structurally separable, non-linear models can significantly sharpen tail separation.

---

## 5. Representation Matters More Than Hyperparameters

Isolation Forest tuning produced only marginal improvements compared to:

- Feature engineering
- Seasonality handling
- Dataset characteristics

On machine temperature:

- ML reduced alert rate substantially relative to MAD.
- Hyperparameter adjustments provided incremental gains.
- Feature representation dictated performance ceiling.

On taxi:

- Even tuned ML did not materially outperform robust statistical baselines.

**Insight:**
Model architecture contributes less than signal representation and anomaly structure.

---

## 6. Point-Wise Evaluation Is Harsh But Informative

Exact timestamp matching:

- Penalizes region-level detections.
- Understates performance in window-based anomaly definitions.

However:

- Near-hit evaluation did not materially change taxi results.
- Machine temperature retained high event recall under window evaluation.

This confirms that:

- Taxi anomalies were genuinely hard to isolate.
- Machine temperature anomalies were structurally distinct.

---

## 7. Dataset-Dependent ML Value

Across both datasets:

| Dataset             | Statistical Baseline                     | ML Impact                           |
| ------------------- | ---------------------------------------- | ----------------------------------- |
| NYC Taxi            | Low recall under reasonable alert budget | Minimal improvement                 |
| Machine Temperature | High recall but high alert volume        | ~3× alert reduction at equal recall |

**Core Takeaway:**

Machine learning is not universally superior to statistical baselines.

Its value depends on:

- Signal-to-noise structure
- Anomaly separability
- Feature representation
- Operational alert constraints

---

# Engineering Conclusion

This comparative study demonstrates:

- The importance of seasonality handling
- The role of robust statistics
- The primacy of alert-budget evaluation
- The dataset-dependent benefit of ML
- The limits of hyperparameter tuning
- The importance of anomaly ranking quality

The seasonal-differenced rolling MAD model serves as a credible statistical baseline.

Isolation Forest demonstrates meaningful operational gains when anomaly structure is separable.

Together, these results provide a realistic framework for evaluating anomaly detection systems beyond headline metrics.
