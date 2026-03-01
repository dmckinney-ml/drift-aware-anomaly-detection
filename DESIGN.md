## 1. Problem Definition

Modern production systems generate continuous streams of operational metrics such as API latency, transaction volume, CPU utilization, sales counts, and request throughput. Undetected anomalies in these signals can indicate system degradation, outages, fraud, infrastructure instability, or unexpected business events.

The objective of this project is to design and implement a drift-aware anomaly detection system capable of:

- Detecting anomalous behavior in univariate time-series data
- Comparing statistical and ML-based detection approaches
- Supporting configurable alert thresholds
- Monitoring distribution shift over time
- Providing a retraining strategy when drift is detected

The system should be deployable as a lightweight service and usable by the public via CSV upload or streaming simulation.

## 2. Business Impact Framing

Operational anomaly detection is critical in environments where:

- High latency impacts customer experience
- Unexpected sales drops affect revenue
- Infrastructure instability creates outages
- Silent metric drift reduces model reliability

A well-calibrated anomaly detection system must balance:

- False Positives (alert fatigue, wasted investigation time)
- False Negatives (missed incidents, business loss)

The value of this system lies in:

- Early detection of abnormal behavior
- Controlled alert volume via threshold tuning
- Detection of long-term distribution shift (concept drift)
- Clear evaluation of tradeoffs between baseline statistical methods and ML-based approaches

## 3. Baseline Approach: Statistical Thresholding

Before applying ML, a statistical baseline is implemented.

### Rolling Z-Score Detector

For a sliding window of size W:

- Compute rolling mean μ
- Compute rolling standard deviation σ
- Calculate z-score:

  $z = \frac{x - \mu}{\sigma}$

An observation is flagged as anomalous if:

    $|z| > \text{threshold}$

### Robust Alternative (Optional)

To reduce sensitivity to outliers:

- Use rolling median
- Use Median Absolute Deviation (MAD)

#### Why Baseline First?

Senior reasoning:

- Establish minimum viable detection capability
- Provide interpretability
- Create a benchmark for ML comparison
- Detect whether ML meaningfully improves performance

## 4. ML Candidate Models

To compare against the baseline:

### Isolation Forest (Primary ML Model)

- Unsupervised anomaly detection
- Works well for high-dimensional or non-Gaussian data
- Handles irregular distributions better than z-score

**Input features:**

- Current value
- Rolling statistics
- Optional lag features

**Output:**

- Anomaly score
- Binary anomaly label via thresholding

#### Model Selection Rationale

Isolation Forest was selected because:

- No labeled data is required
- Efficient for moderate dataset sizes
- Suitable for operational anomaly detection

Neural networks were intentionally not selected due to:

- Higher complexity
- Greater training instability
- Limited incremental value for simple univariate signals

## 5. Evaluation Metrics

Evaluation depends on availability of labeled anomalies.

### With Labels

- Precision
- Recall
- F1 Score
- PR-AUC (preferred over ROC for imbalanced anomalies)
- Alert rate (percentage of points flagged)

#### Threshold Strategy

Thresholds may be:

- Fixed (e.g., z > 3)
- Quantile-based (e.g., top 0.5% anomaly scores)

Threshold selection must consider:

- Acceptable alert volume
- Cost of false positives
- Cost of missed anomalies

### Without Labels

- Alert stability over time
- Score distribution analysis
- Manual inspection sampling

## 6. Production Architecture

### Data Flow

- Ingest time-series data (CSV or streaming simulation)
- Preprocess and sort by timestamp
- Generate rolling features
- Apply anomaly detection model
- Log anomaly scores and decisions
- Store results (e.g., BigQuery or in-memory storage)
- Expose via REST API (Cloud Run container)

### Deployment

- Containerized with Docker
- Deployable to Cloud Run
- Exposes REST endpoint for inference
- Hugging Face Space for public demo (Gradio UI)

Architecture emphasizes:

- Stateless inference
- Configurable window and threshold parameters
- Scalability via serverless compute

## 7. Monitoring Strategy

Operational ML systems require continuous monitoring.

**Metrics to track:**

- Prediction latency
- Error rate
- Anomaly rate (alerts per time window)
- Distribution of anomaly scores

Alerting can be triggered if:

- Anomaly rate exceeds expected threshold
- Prediction latency spikes
- Model errors increase

## 8. Drift Detection Plan

Model performance may degrade over time due to:

- Seasonal changes
- User behavior shifts
- Infrastructure evolution

Drift is detected by comparing:

Reference Window Distribution
vs
Recent Window Distribution

Using:

- Population Stability Index (PSI)
- KL Divergence

Drift levels categorized as:

- Low
- Moderate
- High

High drift triggers retraining evaluation.

## 9. Retraining Policy

Retraining may be triggered by:

- Sustained high drift
- Degraded precision/recall
- Alert volume instability

Retraining process:

- Collect new data window
- Recompute rolling features
- Retrain Isolation Forest
- Validate against holdout set
- Compare performance to previous model
- Promote model if performance improves
- Rollback supported via versioning.

## 10. Failure Modes

Potential system limitations:

- Strong seasonality may cause false positives
- Level shifts may appear as sustained anomalies
- Highly noisy signals may inflate alert rate
- Non-stationary variance can degrade z-score performance
- Isolation Forest sensitivity to contamination parameter

Mitigations include:

- Seasonal decomposition
- Adaptive window sizing
- Robust scaling
- Regular drift monitoring
