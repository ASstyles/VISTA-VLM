# VISTA-VLM v4.0 Evaluation Protocol & Metric Standards

## 1. Multi-Track Evaluation Philosophy

To prevent misleading metric inflation and maintain absolute research rigor, VISTA-VLM strictly separates benchmark evaluation into four independent tracks:

### Track 1: In-Domain Normal-Only Industrial Inspection
- **Premise**: Target category normal reference images are available for training/fitting.
- **Competitors**: PatchCore, PaDiM, VISTA-VLM (in-domain).
- **Goal**: Measure maximum defect localization fidelity and image discrimination on the target production line.

### Track 2: Source-Only Cross-Category Transfer
- **Premise**: Model is trained exclusively on a single source category (`bottle`) and evaluated zero-shot across distinct target categories (`cable`, `capsule`, `leather`, `tile`, `candle`) without fine-tuning.
- **Competitors**: VISTA-VLM (Source: Bottle).
- **Goal**: Quantify cross-domain geometric defect generalization.

### Track 3: Zero-Shot Multimodal VLM
- **Premise**: Zero target normal training images. Multimodal alignment using language prompt ensembles.
- **Competitors**: WinCLIP, AnomalyCLIP.
- **Goal**: Measure open-vocabulary zero-shot inspection capability without any exemplar images.

### Track 4: Few-Normal-Shot Sample Efficiency
- **Premise**: Strict sample constraints with $k \in \{1, 2, 4, 8\}$ normal reference images.
- **Competitors**: PatchCore, PaDiM.
- **Goal**: Quantify rapid adaptation efficiency in low-data manufacturing lines.

---

## 2. Standard Metric Definitions

### 2.1 Image-Level Metrics
- **AUROC (Area Under ROC Curve)**: Evaluates ranking quality across all classification thresholds:
  $$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$$
- **Average Precision (AP / PR-AUC)**: Measures precision across all recall levels, especially robust under severe class imbalance (normal vs rare defects).
- **F1 Score**: Harmonic mean of precision and recall evaluated at calibrated decision threshold $\tau_c$.

### 2.2 Pixel-Level Metrics
- **Pixel AUROC**: Area under the receiver operating characteristic computed across all pixels in the test evaluation set.
- **Pixel AP**: Precision-recall area for pixel-level anomaly localization.
- **AUPRO (Area Under Per-Region Overlap)**:
  Measures overlap for each connected defect component up to a maximum false-positive rate limit ($\text{FPR} \le 0.30$), preventing large defects from dominating small-defect localization metrics:
  $$\text{AUPRO} = \frac{1}{0.3} \int_0^{0.3} \left( \frac{1}{K} \sum_{k=1}^K \frac{|M_{\text{pred}} \cap R_k|}{|R_k|} \right) d\text{FPR}$$

### 2.3 Reliability & Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures calibration gap across $M=10$ confidence bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Empirical False Alarm Rate**: Fraction of nominal test samples incorrectly classified as anomalous at significance level $\alpha = 0.05$.
- **Wilson Score 95% Confidence Interval**: Rigorous non-asymptotic confidence bounds on observed false-pass rates.

---

## 3. Execution Commands

```bash
# Run complete 4-track benchmark across all 6 categories
python scripts/run_multi_track_benchmark.py

# Run systematic 10-configuration ablation study
python vista_vlm/experiments/run_ablation.py

# Run failure taxonomy and error analysis
python scripts/run_error_analysis.py
```

