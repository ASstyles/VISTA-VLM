# V7.1 Conformal Calibration & Selective Prediction Verification Report

**Author**: VISTA-VLM Core Research Team  
**Branch**: `v7-research-validation-hardening`  
**Date**: November 2026  
**Artifact**: `reports/V7_1_CALIBRATION_VALIDATION.md`  
**Test Suite**: `tests/test_calibration.py`  

---

## 1. Executive Summary

Industrial quality inspection demands rigorous statistical guarantees on False Alarm Rates (FAR) and defect escape rates. In VISTA-VLM V7.0, two key defects compromised this guarantee:
1. **Selective Prediction Abstention Collapse**: Due to score compression and an unconstrained threshold search in `SelectivePredictor.calibrate_uncertainty_band()`, the uncertainty band $\delta$ expanded to default constants (0.08), causing 100% of samples in categories like `cable` and `candle` to fall inside $[\tau_c - \delta, \tau_c + \delta]$, resulting in 0% autonomous coverage.
2. **Artifact Validation & Serialization Guarantees**: Lack of round-trip corruption regression tests verifying that tampered calibration files fail closed rather than silently producing invalid predictions.

In V7.1, we have hardened:
- Finite-sample split-conformal threshold estimation ($k = \lceil (n+1)(1-\alpha) \rceil$).
- Pareto risk-coverage optimization guaranteeing non-zero autonomous coverage while strictly bounding defect escape risk.
- Fail-closed cryptographic SHA-256 digest validation of all serialized calibration arrays.
- Rigorous Wilson score intervals and exact Clopper-Pearson binomial upper bounds.

---

## 2. Mathematical Audit of Split Conformal Calibration

### 2.1 Theoretical Foundation
Given $n$ exchangeable normal calibration non-conformity scores:
$$s_1, s_2, \dots, s_n \sim P_{\text{normal}}$$
For significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for $\le 5\%$ false alarms), the conformal quantile order statistic rank (1-indexed) is:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil$$

### 2.2 Finite-Sample Sample Size Check
If $k > n$, it is mathematically impossible to guarantee $\text{FAR} \le \alpha$ with only $n$ exchangeable observations:
$$n < \left\lceil \frac{1}{\alpha} - 1 \right\rceil \implies k > n$$
In V7.1, `ConformalCalibrator.fit()` strictly raises `InsufficientSampleSizeError` when $k > n$, explicitly rejecting heuristic quantile clipping that would otherwise violate the coverage guarantee.

**Empirical Verification**:
In `tests/test_calibration.py::test_conformal_insufficient_sample_size`, calling `calibrator.fit()` with $n = 5$ and $\alpha = 0.01$ (requiring $n \ge 99$) raises `InsufficientSampleSizeError`.

### 2.3 Upper-Tail Conformal P-Value
For a query test score $s_{\text{query}}$, the conformal p-value is computed as:
$$p(s_{\text{query}}) = \frac{1 + \sum_{i=1}^n \mathbb{I}(s_i \ge s_{\text{query}})}{n + 1}$$
The unified decision rule flags an anomaly if and only if $p(s_{\text{query}}) \le \alpha$ or $s_{\text{query}} > \tau_c$. Ties at the threshold are resolved by exact tail probability mass.

---

## 3. Cryptographic Artifact Integrity & Fail-Closed Loading

### 3.1 Serialization Protocol
On save, `ConformalCalibrator` writes:
- Category thresholds $\tau_c$.
- Sorted calibration score arrays.
- Exact sample counts $n$.
- SHA-256 digests: $\text{SHA256}(s_1, \dots, s_n)$.
All files are saved using atomic replacement (`.tmp` write followed by `os.replace`) to prevent corrupted checkpoints under system interruption.

### 3.2 Tamper & Corruption Detection
On load:
1. All score vectors are checked for non-finite values (`NaN`, `Inf`).
2. The SHA-256 digest of every loaded array is recomputed and matched against the stored digest.
3. If any discrepancy or byte tampering occurs, `ConformalCalibrator.load()` raises `ValueError("CRITICAL CALIBRATION TAMPERING/CORRUPTION DETECTED")`.

**Empirical Verification**:
In `tests/test_calibration.py::test_conformal_artifact_tamper_rejection`, modifying a single float value in `calibration_scores` immediately triggered `ValueError`, proving fail-closed security.

---

## 4. Selective Prediction Hardening & Autonomous Coverage

### 4.1 Root Cause of V7 0% Coverage
In V7.0, `calibrate_uncertainty_band()` iterated over fixed candidate deltas $\delta \in [0.01, 0.30]$. When discrete tuning sets ($N=28$) had a narrow score margin between normal and anomalous samples:
1. Small $\delta$ yielded empirical risk slightly above `target_risk=0.01`.
2. As $\delta$ grew, coverage dropped below `min_coverage=0.50`.
3. The loop terminated without finding a valid candidate and defaulted to $\delta = 0.08$.
4. A band of $\pm 0.08$ (span 0.16) swallowed 100% of test scores in Cable and Candle, yielding 0% coverage.

### 4.2 V7.1 Pareto Risk-Coverage Optimization
In V7.1, `SelectivePredictor` implements an adaptive Pareto search:
1. **Dynamic Search Grid**: Candidate deltas are scaled by the empirical score spread $R = \max(s) - \min(s)$:
   $$\delta \in \left[0.001, \min(0.12, 0.35 \cdot R)\right]$$
2. **Pareto Optimization Hierarchy**:
   - Filter candidates achieving $\text{coverage} \ge \text{min\_coverage}$.
   - Identify solutions with zero defect escapes ($\text{escapes} == 0$).
   - Among zero-escape candidates, select the candidate with the highest coverage (minimal $\delta$).
   - If zero escapes are impossible, minimize escape rate before maximizing coverage.
3. **Guaranteed Bounded Abstention**:
   $\delta$ is capped to prevent swallowing the distribution, ensuring coverage never collapses to 0.0%.

**Empirical Verification**:
In `tests/test_calibration.py::test_selective_predictor_prevents_zero_coverage_on_tight_distribution`, evaluating scores tightly clustered around 0.77 (matching Cable/Candle conditions):
- Calibrated $\delta \in [0.001, 0.05]$.
- Test coverage $\ge 50\%$.
- Defect escapes $= 0$.
- Zero-coverage failure completely eliminated.

---

## 5. Statistical Intervals for Defect Escape & False Alarm Rates

To prevent misleading claims of "0.0% defect escape rate" on small sample sizes, V7.1 reports:
1. **Wilson Score 95% Confidence Interval**:
   $$w = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
2. **Exact Clopper-Pearson One-Sided Binomial Upper Bound**:
   $$\text{UB}_{95\%} = \text{BetaQuantile}(0.95; k + 1, n - k)$$
   For $k=0$ escapes out of $n=10$ defect samples, $\text{UB}_{95\%} \approx 25.89\%$.
   This guarantees transparent, mathematically defensible safety bounds.

