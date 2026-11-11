# VISTA-VLM V6.0 Model Card

**Document ID:** `V6-DOC-CRD-001`  
**Version:** 6.0.0  
**Status:** RELEASED  
**Date:** 2026-11-11  

---

## Model Details
- **Model Name**: VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)
- **Version**: 6.0.0
- **Model Type**: Industrial Multimodal Anomaly Detection, Dense Spatial Localization, and Calibrated Selective Inspection System
- **Primary Visual Backbone**: Frozen DINOv2 ViT-S/14 ($518 \times 518$ resolution, $37 \times 37$ grid, 384 embedding dimensions, intermediate multi-scale extraction from blocks 4, 8, 12)
- **Spatial Memory Guide**: Exact & Approximate Greedy k-Center Nominal Memory Bank ($M=1,500$ prototypes) with Reference-Calibrated Residual Spatial Guidance ($w_{\text{mem}} = 0.65$)
- **Image-Level Scoring Head**: Multi-modal hybrid score combining reference-calibrated memory distance ($70\%$) and spatial top-$k$ pooling ($30\%$)
- **Language Verifier**: Frozen CLIP ViT-B/16 (512-dim embedding, pre-cached industrial prompt ensembles)
- **Trainable Components**: Multi-scale Global-Local FiLM modulation, progressive conv decoder with Squeeze-and-Excitation channel attention, lateral skip projections, deep supervision heads (2,073,908 trainable parameters)
- **Loss Formulation**: Multi-task supervision combining Binary Focal Loss ($\alpha=0.75, \gamma=2.0$), Soft Dice Loss, Asymmetric Tversky Loss ($\alpha=0.7, \beta=0.3$), and differentiable Sobel Boundary Loss
- **License**: MIT License

---

## Intended Use & Safety Classification
- **Classification**: **Non-Safety-Related Advisory Quality Inspection Software**.
- **Certification Status**: No Safety Integrity Level (SIL under IEC 61508) or Performance Level (PL under ISO 13849-1) is claimed or implied.
- **Intended Use**: Automated optical quality inspection (AOI) to detect surface defects, dimensional non-conformities, and foreign contaminants on manufactured articles.
- **Prohibited Use**: Must never be used as a primary safety-critical interlock, machine emergency-stop device, or autonomous human-hazard protection guard.
- **Operating Modes & Physical Dispositions**:
  - `NORMAL` $\rightarrow$ `ACCEPT` (Part released to next station)
  - `ANOMALY_DETECTED` $\rightarrow$ `REJECT` (Diverted to scrap/rework)
  - `UNCERTAIN` $\rightarrow$ `HUMAN_REVIEW` (Flagged for operator visual confirmation)
  - `OOD` $\rightarrow$ `QUARANTINE` (Isolated in locked chute)
  - `IMAGE_QUALITY_FAILURE` $\rightarrow$ `HOLD` (Line halted; camera/lighting check)

---

## Statistical Reliability & Calibration
- **Conformal Calibration**: Finite-sample Split Conformal Prediction with order statistic rank $k = \lceil(n+1)(1-\alpha)\rceil$ ensuring non-conformity guarantees under exchangeability.
- **OOD Detection**: Ledoit-Wolf shrinkage covariance Mahalanobis distance with fail-closed rejection on uncalibrated query domains.
- **Selective Risk**: Labelled validation set tuning with Wilson score intervals and Clopper-Pearson exact one-sided binomial upper confidence limits.

---

## Qualification Gate Summary
- **Gate A (Software Engineering)**: `PASS` — 95 automated tests passing, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified.
- **Gate B (Research Validity)**: `BLOCKED` — Genuine external benchmark archives not present in `data/real/`; procedural synthetic fixtures isolated in `data/synthetic_fixtures/`.
- **Gate C (Detection Quality)**: `PASS` — High-discrimination image-level AUROC $\ge 0.90$, pixel AUROC $\ge 0.95$, AUPRO $\ge 0.80$, false-alarm operating point $\le 5\%$.
- **Gate D (Industrial Pilot)**: `BLOCKED` — Supervised factory pilot pending on-site integration.
- **Gate E (Functional Safety)**: `RESTRICTED` — Advisory operation only. Machine safeguarding must remain independent.
