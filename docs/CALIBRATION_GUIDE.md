# VISTA-VLM V5.0 Statistical Calibration and OOD Fitting Guide

**Document ID:** `V5-DOC-CAL-001`  
**Version:** 5.0.0  
**Target:** Machine Learning Engineers & Quality Specialists  
**Classification:** TECHNICAL REFERENCE & CALIBRATION MANUAL  

---

## 1. Overview of Statistical Calibration

Standard machine learning models output raw anomaly scores (e.g. $[0, 1]$ or Euclidean distances) that do not represent statistically valid probabilities. In VISTA-VLM V5.0, predictions are governed by two rigorous statistical mechanisms:
1. **Split Conformal Calibration**: Quantifies uncertainty with finite-sample statistical guarantees on false alarm bounds ($\alpha$).
2. **Mahalanobis OOD Detection**: Detects foreign objects, lighting failures, or wrong part batches by modeling nominal feature covariance.

---

## 2. Step-by-Step Calibration CLI Workflow

### Step 1: Prepare Normal Calibration Images
Collect at least $N = 20$ (recommended $N = 50$) verified normal, defect-free parts from the production line:
```
data/calibration/bottle/
  sample_001.png
  sample_002.png
  ...
  sample_050.png
```

### Step 2: Fit Nominal Memory Bank and Conformal Quantiles
Execute the unified calibration pipeline:
```bash
python scripts/train_and_calibrate.py \
  --category bottle \
  --data-dir data/calibration \
  --alpha 0.05 \
  --output-dir checkpoints/
```

This generates two cryptographic artifacts:
- `checkpoints/calibration_bottle.json`: Contains the conformal non-conformity scores and the calibrated threshold $\hat{q}$ at $1 - \alpha = 0.95$ confidence.
- `checkpoints/ood_bottle.json`: Contains the mean feature vector $\mu$ and inverted covariance matrix $\Sigma^{-1}$ across nominal DINOv2 tokens.

---

## 3. Selecting Significance Level ($\alpha$)

The significance parameter $\alpha$ represents the **tolerable false alarm rate on normal items**:
- **$\alpha = 0.05$ (Standard Industrial)**: Guarantees $\le 5\%$ false alarm rate on nominal production.
- **$\alpha = 0.01$ (High-Throughput Packaging)**: Minimizes false alarms ($\le 1\%$) when scrap cost is very high. Requires $N \ge 100$ calibration samples to accurately estimate the 99th percentile.
- **$\alpha = 0.10$ (Strict Defect Catch)**: Maximizes defect sensitivity for high-risk components.

---

## 4. Interpreting Output Reliability Indicators

Inference responses return structured `reliability` telemetry:
```json
{
  "reliability": {
    "is_calibrated": true,
    "operational_state": "NORMAL",
    "disposition": "ACCEPT",
    "p_value": 0.8421,
    "calibrated_threshold": 0.5000,
    "ood_distance": 0.4210,
    "is_ood": false
  }
}
```

- **`p_value`**: The empirical probability of observing a non-conformity score as high or higher among nominal samples. High p-values ($> 0.05$) indicate typical normal parts. Extremely low p-values ($< 0.001$) indicate statistically significant anomalies.
- **`is_calibrated`**: Must be `true` in production. If `false`, the operational state automatically falls back to `CALIBRATION_INVALID` and physical disposition `QUARANTINE`.

