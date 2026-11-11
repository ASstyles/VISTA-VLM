# VISTA-VLM V5.0 Evaluation Protocol & Metric Standards

**Document ID:** `V5-DOC-EVL-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Multi-Track Evaluation Philosophy

To prevent misleading metric inflation and maintain absolute research rigor, VISTA-VLM strictly separates benchmark evaluation into four independent tracks:

### Track 1: In-Domain Normal-Only Industrial Inspection
- **Premise**: Target category normal reference images are available for training/fitting.
- **Competitors**: PatchCore, PaDiM, VISTA-VLM v5.0.
- **Goal**: Measure maximum defect localization fidelity and image discrimination on the target production line.

### Track 2: Source-Only Cross-Category Transfer
- **Premise**: Model is trained exclusively on a single source category (`bottle`) and evaluated zero-shot across distinct target categories (`cable`, `capsule`, `leather`, `tile`, `candle`) without fine-tuning.
- **Competitors**: VISTA-VLM v5.0 (Source: Bottle).
- **Goal**: Quantify cross-domain geometric defect generalization.

### Track 3: Zero-Shot Multimodal VLM
- **Premise**: Zero target normal training images. Multimodal alignment using language prompt ensembles.
- **Competitors**: WinCLIP, AnomalyCLIP.
- **Goal**: Measure open-vocabulary zero-shot inspection capability without any exemplar images.

### Track 4: Few-Normal-Shot Sample Efficiency ($k \in \{1, 2, 4, 8, 16\}$)
- **Premise**: Strict sample constraints with $k$ normal reference images for coreset memory fitting.
- **Competitors**: PatchCore, PaDiM, VISTA-VLM v5.0.
- **Goal**: Quantify rapid adaptation efficiency in low-data manufacturing lines.

---

## 2. Standard Metric Definitions

### 2.1 Image-Level Metrics
- **AUROC (Area Under ROC Curve)**: Evaluates ranking quality across all classification thresholds:
  $$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$$
- **Average Precision (AP / PR-AUC)**: Measures precision across all recall levels, especially robust under severe class imbalance.
- **$F_1$ Score**: Harmonic mean of precision and recall evaluated at calibrated decision threshold $\tau_c$.

### 2.2 Pixel-Level Metrics & Exact Quantile AUPRO
- **Pixel AUROC**: Area under the ROC curve computed across all pixels in the test evaluation set.
- **Pixel AP**: Precision-recall area for pixel-level anomaly localization.
- **AUPRO (Area Under Per-Region Overlap)**:
  Measures overlap for each connected defect component up to a maximum false-positive rate limit ($\text{FPR} \le 0.30$), preventing large defects from dominating small-defect localization metrics:
  $$\text{AUPRO} = \frac{1}{\text{max\_fpr}} \int_0^{\text{max\_fpr}} \left( \frac{1}{K} \sum_{k=1}^K \frac{|M_{\text{pred}} \cap R_k|}{|R_k|} \right) d\text{FPR}$$
  *Implementation Note*: Computed via exact trapezoidal integration over non-uniform quantile threshold sweeps.

### 2.3 Defect Size Stratification
To expose model sensitivity across varying physical defect scales, recall and mean overlap are stratified across 4 bins:
- **Tiny**: Area $< 0.1\%$ of total image area.
- **Small**: Area $0.1\% - 1.0\%$.
- **Medium**: Area $1.0\% - 5.0\%$.
- **Large**: Area $> 5.0\%$.

### 2.4 Reliability, Calibration & Statistical Risk Bounds
- **Expected Calibration Error (ECE)**: Measures calibration gap across $M=10$ confidence bins.
- **Wilson Score 95% Confidence Interval**:
  $$w = \frac{k + \frac{z^2}{2}}{n + z^2} \pm \frac{z}{n + z^2} \sqrt{\frac{k(n - k)}{n} + \frac{z^2}{4}}$$
- **Rule of 300**: Proves that certifying false escape rate $< 1.0\%$ at $95\%$ confidence requires testing at least 300 verified defective parts with zero escapes.
