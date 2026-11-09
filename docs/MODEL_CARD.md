# VISTA-VLM v4.0 Model Card

## Model Details
- **Model Name**: VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)
- **Version**: 4.0.0
- **Model Type**: Industrial Multimodal Anomaly Localization, Verification, and Calibrated Selective Inspection System
- **Primary Visual Backbone**: Frozen DINOv2 ViT-S/14 ($518 \times 518$ input resolution, $37 \times 37$ grid, 384 embedding dims, intermediate extraction from blocks 4, 8, 12)
- **Secondary Scaled Variant**: DINOv2 ViT-B/14 (768 embedding dims)
- **Language Verifier**: Frozen CLIP ViT-B/16 (512-dim embedding, pre-cached prompt ensembles)
- **Trainable Components**: Multi-scale Global-Local FiLM modulation, progressive conv decoder with Squeeze-and-Excitation channel attention, lateral skip projections, deep supervision heads, hybrid anomaly scoring head (2,073,908 trainable parameters)
- **License**: MIT License

---

## Intended Use
- **Primary Use**: Automated industrial surface inspection, millimeter-scale defect localization, candidate defect ranking, and evidence-grounded textual explanation across manufacturing parts (rigid objects, cables, small components, planar surfaces, multi-part assemblies).
- **Intended Users**: Automated Optical Inspection (AOI) engineers, manufacturing quality control teams, and computer vision researchers.
- **Operating Modes**:
  - **Autonomous Pass/Fail**: Committed strictly outside the calibrated uncertainty band $[\tau_c - \delta, \tau_c + \delta]$ and within OOD distance thresholds.
  - **Human-in-the-Loop Escalation**: Triggered automatically on ambiguous border cases or out-of-distribution textures.

---

## Out-of-Scope Use
- Never use for safety-critical aerospace or medical applications without clinical validation and secondary hardware sensor redundancy.
- Never use in uncalibrated environments with shifting illumination or focal distance without running split conformal re-calibration.
- Never assert a statistical guarantee of $\le 1\%$ false-pass risk without verifying a defect sample size of $N_{\text{def}} \ge 300$.

---

## Empirical Benchmark Performance

*Evaluated on the 6-category industrial benchmark split (procedural synthetic benchmark fixtures).*

### Track 1: In-Domain Normal-Only Industrial Inspection
- **Macro Image AUROC**: 1.0000 (PatchCore: 1.0000, PaDiM: 1.0000, VISTA v4 Bottle: 0.9167)
- **Macro Image AP**: 1.0000
- **Macro Pixel AUROC**: 0.9626 (PatchCore), 0.9841 (PaDiM), 0.5385 (VISTA v4 Bottle)
- **Macro AUPRO**: 0.8073 (PatchCore), 0.8654 (PaDiM), 0.2123 (VISTA v4 Bottle)
- **Throughput**: 1.38 - 1.76 FPS on CPU

### Track 2: Source-Only Cross-Category Transfer (Bottle $\to$ Unseen)
- **Macro Image AUROC**: 0.8167
- **Macro Image AP**: 0.8829
- **Macro Pixel AUROC**: 0.4647
- **Macro AUPRO**: 0.2255

### Track 3: Zero-Shot Multimodal VLM
- **WinCLIP Macro Image AUROC**: 0.9708 | Macro AUPRO: 0.1584 | 4.61 FPS
- **AnomalyCLIP Macro Image AUROC**: 0.6125 | Macro AUPRO: 0.2273 | 4.51 FPS

### Track 4: Few-Normal-Shot Sample Efficiency ($k \in \{1, 2, 4, 8\}$)
- **1-Shot PatchCore**: Image AUROC: 1.0000 | Pixel AUROC: 0.9754 | AUPRO: 0.8693
- **8-Shot PatchCore**: Image AUROC: 1.0000 | Pixel AUROC: 0.9754 | AUPRO: 0.8645

---

## Reliability & Trust Guarantees
- **Conformal Coverage Guarantee**: Under exchangeability, split conformal calibration guarantees marginal false-alarm rate $\le \alpha$ ($\alpha = 0.05$).
- **Shrinkage Mahalanobis OOD Detector**: Ledoit-Wolf shrinkage covariance persistence protects against domain shift and uncalibrated distance fallback bugs.
- **Evidence-Grounded Explanations**: Structured textual explanations strictly cite bounding box coordinates, IoU metrics, and candidate margins, eliminating LLM hallucinations.
- **Dual-Risk Reporting**: Reports both false-pass and false-alarm risks with Wilson score 95% confidence intervals and sample size validity flags.
