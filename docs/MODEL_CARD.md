# VISTA-VLM V5.0 Model Card

**Document ID:** `V5-DOC-CRD-001`  
**Version:** 5.0.0  
**Status:** RELEASED  
**Date:** 2026-11-11  

---

## Model Details
- **Model Name**: VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)
- **Version**: 5.0.0
- **Model Type**: Industrial Multimodal Anomaly Detection, Dense Spatial Localization, and Calibrated Selective Inspection System
- **Primary Visual Backbone**: Frozen DINOv2 ViT-S/14 ($518 \times 518$ resolution, $37 \times 37$ grid, 384 embedding dimensions, intermediate multi-scale extraction from blocks 4, 8, 12)
- **Spatial Memory Guide**: Coreset nominal memory bank ($M=1,500$ prototypes) with memory-guided spatial residual fusion ($w_{\text{mem}} = 0.5$)
- **Language Verifier**: Frozen CLIP ViT-B/16 (512-dim embedding, pre-cached prompt ensembles)
- **Trainable Components**: Multi-scale Global-Local FiLM modulation, progressive conv decoder with Squeeze-and-Excitation channel attention, lateral skip projections, deep supervision heads (2,073,908 trainable parameters)
- **License**: MIT License

---

## Intended Use & Safety Classification
- **Classification**: **Advisory / Supervised Quality Inspection Tool (SIL 0)**.
- **Intended Use**: Automated surface inspection, millimeter-scale defect localization, candidate defect ranking, and evidence-grounded textual explanation.
- **Prohibited Use**: Must never be used as a primary safety-critical interlock or autonomous human-hazard protection guard.
- **Operating Modes**:
  - `NORMAL` $\rightarrow$ `ACCEPT` (Part released to next station)
  - `ANOMALY_DETECTED` $\rightarrow$ `REJECT` (Diverted to scrap/rework)
  - `UNCERTAIN` $\rightarrow$ `HUMAN_REVIEW` (Flagged for operator visual confirmation)
  - `OOD` / `IMAGE_QUALITY_FAILURE` $\rightarrow$ `QUARANTINE` / `HOLD`

---

## Empirical Benchmark Performance

*Evaluated on standardized industrial benchmark fixtures conforming to MVTec AD and VisA directory schemas.*

### Track 1: In-Domain Normal-Only Industrial Inspection
- **Bottle Pixel AUROC**: **$0.9966$** (PatchCore: $0.9980$, PaDiM: $0.9982$)
- **Bottle AUPRO ($FPR \le 0.30$)**: **$0.9713$** (PatchCore: $0.9846$, PaDiM: $0.9845$)
- **Bottle Tiny Defect Recall ($<0.1\%$ area)**: **$83.64\%$** (PatchCore: $16.14\%$, PaDiM: $21.82\%$)
- **Macro Pixel AUROC (6 Categories)**: **$0.9740 \pm 0.0256$**
- **Throughput (CPU)**: $1.45 - 1.68\text{ FPS}$

### Track 2: Source-Only Cross-Category Transfer (Bottle $\to$ Unseen)
- **Macro Image AUROC**: **$0.7433 \pm 0.1525$**
- **Macro Image AP**: **$0.8281$**
- **Macro Pixel AUROC**: **$0.8805 \pm 0.0950$**
- **Macro AUPRO**: **$0.5304$**

### Track 3: Zero-Shot Multimodal VLM
- **WinCLIP Macro Pixel AUROC**: $0.5163 \pm 0.2675$ | Macro AUPRO: $0.1597$
- **AnomalyCLIP Macro Pixel AUROC**: $0.5547 \pm 0.2071$ | Macro AUPRO: $0.2119$

### Track 4: Few-Normal-Shot Sample Efficiency (Bottle)
- **1-Shot Pixel AUROC**: **$0.9969$** (PatchCore: $0.9970$, PaDiM: $0.9881$)
- **2-Shot Pixel AUROC**: **$0.9951$** (PatchCore: $0.9980$, PaDiM: $0.9955$)
- **4-Shot Pixel AUROC**: **$0.9951$** (PatchCore: $0.9982$, PaDiM: $0.9983$)
- **8-Shot Pixel AUROC**: **$0.9963$** (PatchCore: $0.9984$, PaDiM: $0.9984$)
- **16-Shot Pixel AUROC**: **$0.9968$** (PatchCore: $0.9985$, PaDiM: $0.9985$)

---

## Provenance and Traceability Disclosure
- Benchmark artifacts: `benchmarks/results/multi_track_benchmark.json`
- Synthetic manifests: `data/manifests/synthetic/mvtec_like_manifest.json`
- Evaluation engine: `scripts/run_multi_track_benchmark.py`
