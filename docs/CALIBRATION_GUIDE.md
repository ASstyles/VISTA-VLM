# VISTA-VLM V6.0 Statistical Calibration & Conformal Prediction Guide

**Document ID:** `V6-DOC-CAL-001`  
**Version:** 6.0.0  
**Target:** Statistical Learning Engineers & Metrology Specialists  
**Classification:** TECHNICAL REFERENCE & CALIBRATION MANUAL  

---

## 1. Mathematical Principles of Split Conformal Calibration

Standard deep learning models output raw scalar scores that lack rigorous frequentist or Bayesian guarantees. VISTA-VLM V6.0 implements mathematically sound **Finite-Sample Split Conformal Prediction** (Vovk et al., 2005; Angelopoulos & Bates, 2021).

### 1.1 Non-Conformity Score Formulation
Given a calibration set of $n$ independent, identically distributed normal (in-control) instances $\mathcal{D}_{\text{cal}} = \{x_1, \dots, x_n\}$, we evaluate non-conformity scores:
$$s_i = S(x_i) \quad \text{for } i \in \{1, \dots, n\}$$
where $S(x)$ is VISTA's hybrid anomaly scoring function.

### 1.2 Exact Finite-Sample Order Statistic
To guarantee that the marginal false-alarm rate on future exchangeable normal items is bounded by $\alpha \in (0, 1)$:
$$\mathbb{P}(S(X_{n+1}) > \tau_{\text{conf}}) \le \alpha$$
the calibrated decision threshold $\tau_{\text{conf}}$ must be chosen as the $k$-th smallest value (1-indexed order statistic) of the calibration scores:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil$$
$$\tau_{\text{conf}} = s_{(k)}$$

> [!IMPORTANT]
> **Sample Size Requirement & Strict Rejection**:  
> If $k > n$, the requested significance level $\alpha$ is mathematically unsupported by the available calibration sample size $n$. In V6.0, the system raises `InsufficientSampleSizeError` and refuses to emit an arbitrary or uncalibrated threshold. For example, at $\alpha = 0.05$, a minimum of $n = 19$ samples is required ($k = \lceil 20 \times 0.95 \rceil = 19 \le 19$).

### 1.3 Upper-Tail Conformal p-Value
For any query sample $x_{\text{test}}$ with score $s_{\text{test}} = S(x_{\text{test}})$, VISTA-VLM computes the exact conformal p-value:
$$p(x_{\text{test}}) = \frac{1 + \sum_{i=1}^n \mathbb{I}(s_i \ge s_{\text{test}})}{n + 1}$$
Under the null hypothesis that $x_{\text{test}}$ is drawn from the nominal normal distribution (and assuming exchangeability with calibration data), $p(x_{\text{test}})$ is super-uniformly distributed on $[0, 1]$. An alarm is triggered if and only if:
$$p(x_{\text{test}}) \le \alpha \iff s_{\text{test}} \ge \tau_{\text{conf}}$$

### 1.4 Exchangeability Assumptions & Boundary Conditions
- **Exchangeability**: The calibration samples and subsequent test items must be collected under identical optical, sensor, and environmental conditions.
- **Normal-Only Limitation**: Conformal calibration on normal-only data controls the **false-alarm rate (overkill)**. It does **NOT** establish a mathematical guarantee on defect escape (false pass), because no defective instances were present in the calibration null distribution. Defect escape risk is evaluated separately via selective prediction on labelled tuning data.

---

## 2. Regularized Mahalanobis OOD Detection

To detect sensor degradation, lighting loss, or incorrect part categories, VISTA models nominal feature distributions via regularized Mahalanobis distance:
$$D_M(z) = \sqrt{(z - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\text{reg}}^{-1} (z - \boldsymbol{\mu})}$$
where $\boldsymbol{\mu} \in \mathbb{R}^D$ is the empirical mean vector and $\boldsymbol{\Sigma}_{\text{reg}}$ is computed using Ledoit-Wolf shrinkage:
$$\boldsymbol{\Sigma}_{\text{reg}} = (1 - \rho)\boldsymbol{\Sigma}_{\text{emp}} + \rho \frac{\text{Tr}(\boldsymbol{\Sigma}_{\text{emp}})}{D} \mathbf{I}$$
- Prevents singular matrix inversions under rank deficiency ($n < D$).
- OOD threshold $\tau_{\text{OOD}}$ is set to the $98\text{th}$ empirical percentile. Queries exceeding $\tau_{\text{OOD}}$ trigger fail-closed state `OOD` $\to$ `QUARANTINE`.

---

## 3. Labelled Selective Prediction Tuning

To control empirical defect escape on autonomous decisions, VISTA tunes an uncertainty band $[\tau_{\text{conf}} - \delta, \tau_{\text{conf}} + \delta]$ on a **labelled tuning set** containing both normal and defect samples:
- Items inside the uncertainty band are routed to `HUMAN_REVIEW` (Physical disposition: `HOLD`).
- Autonomous `ACCEPT` and `REJECT` decisions occur strictly outside the band.
- Evaluated using **Wilson score 95% confidence intervals** and **Clopper-Pearson exact one-sided binomial upper bounds**:
$$\bar{p}_{\text{exact}} = \text{BetaQuantile}(1 - \alpha_{\text{risk}}, k_{\text{escapes}} + 1, n_{\text{defects}} - k_{\text{escapes}})$$

---

## 4. Calibration CLI Execution

Execute unified training, memory fitting, and statistical calibration:
```bash
python scripts/train_and_calibrate.py \
  --category bottle \
  --epochs 5 \
  --batch_size 4 \
  --alpha 0.05 \
  --output_dir checkpoints/
```

Cryptographic output artifacts:
- `checkpoints/vista_mvtec_bottle.pth`: Model weights + nominal memory bank + category reference stats + SHA-256 digest.
- `checkpoints/calibration_mvtec.json`: Split conformal thresholds, sorted score arrays, and validation hashes.
- `checkpoints/ood_mvtec.json`: Mean vectors, regularized inverted covariance matrices, and distance percentiles.
