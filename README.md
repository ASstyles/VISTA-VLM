# VISTA-VLM: Vision Intelligence through Spatial and Textual Alignment

[![CI](https://github.com/vista-vlm/vista-vlm/actions/workflows/ci.yml/badge.svg)](https://github.com/vista-vlm/vista-vlm/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end industrial visual anomaly detection, localization, and multimodal interpretation framework. VISTA-VLM couples frozen **DINOv2 ViT-S/14** visual representations, **Global-Local FiLM** feature modulation, progressive convolutional segmentation decoders, **CLIP ViT-B/16** semantic defect grounding, and **Split Conformal Reliability Guarantees** with selective prediction gating.

---

## Key Features

- **Frozen Foundation Backbone**: DINOv2 ViT-S/14 ($518 \times 518$ input resolution, $37 \times 37$ spatial grid, 384 embedding dims) with intermediate multi-layer tap at blocks 4, 8, and 12 ($3 \times 384 = 1152$ channels).
- **Global-Local FiLM Fusion**: Feature-wise Linear Modulation $\gamma(z_{cls}) \odot F_{local} + \beta(z_{cls})$ contextually conditions spatial features with global representations.
- **Progressive Convolutional Decoder**: Depthwise-separable convolutions and GroupNorm upsample features through 4 stages ($37 \to 74 \to 148 \to 296 \to 518$).
- **Multimodal Defect Grounding**: Frozen CLIP ViT-B/16 ranks ROI candidate descriptions, identifies semantic defect categories, computes confidence margins, and detects descriptive ambiguity.
- **Conformal Calibration**: Split conformal prediction guarantees finite-sample false-alarm rate control at target $\alpha = 0.05$ on independent nominal calibration splits.
- **Out-of-Distribution & Selective Prediction**: Mahalanobis/Cosine feature prototype OOD detector and conformal uncertainty intervals $[\tau_c - \delta, \tau_c + \delta]$ escalate doubtful predictions to human inspectors.
- **FastAPI Production Engine**: Fully typed OpenAPI endpoints (`/inspect`, `/inspect/batch`, `/health`, `/model/info`, `/benchmark/results`, `/feedback`).
- **VISTA Visual Intelligence Studio**: Production-grade React 19 / TypeScript / Vite interactive dashboard with 5 specialized modules.

---

## System Architecture

```
                             [ Input Image (518x518 RGB) ]
                                          |
                +-------------------------+-------------------------+
                |                                                   |
      [ Frozen DINOv2 ViT-S/14 ]                               [ RGB Crops ]
     Blocks 4, 8, 12 Intermediate                                   |
                |                                                   |
    +-----------+-----------+                                       |
    |                       |                                       |
[Local Patches]       [Global CLS]                                  |
 [1152x37x37]           [1152-dim]                                  |
    |                       |                                       |
    +-----------+-----------+                                       |
                |                                                   |
      [ FiLM Feature Fusion ]                                       |
   γ(z_cls) * F_proj + β(z_cls)                                     |
                |                                                   |
   [ Progressive Conv Decoder ]                                     |
    (37x37 -> 518x518 Upsample)                                     |
                |                                                   |
    +-----------+-----------+                                       |
    |                       |                                       |
[Anomaly Map]       [Image Score]                                   |
 [1x518x518]            [0, 1]                                      |
    |                       |                                       |
    |               +-------+-------------------+                   |
    |               |                           |                   |
    |       [Conformal Calibration]      [OOD Detector]             |
    |         (Quantile τ_c, p-val)     (Prototype Distance)        |
    |               |                           |                   |
    |               +-------------+-------------+                   |
    |                             |                                 |
[Connected Components]  [Selective Prediction]                      |
 (Bounding Boxes / ROIs) (Autonomous vs Human Review)               |
    |                             |                                 |
    +--------------+--------------+                                 |
                   |                                                |
                   v                                                v
    [ Multimodal Verification Engine: Frozen CLIP ViT-B/16 Embeddings ]
                   |
                   v
       [ Structured Inspection Response & Heatmap Overlay ]
```

---

## Genuine Benchmark Results (MVTec AD Bottle)

| Method | Supervision | Image AUROC | Pixel AUROC | Pixel AUPRO | FPS (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VISTA-VLM (Ours)** | **Normal-Only + FiLM + Conformal** | **1.0000** | **0.5625** | **0.6385** | **1.65** |
| WinCLIP | Zero-Shot Text Ensembles | 1.0000 | 0.9031 | 0.7240 | 4.04 |
| PaDiM | Multivariate Gaussian Bank | 1.0000 | 0.8434 | 0.6720 | 1.56 |
| PatchCore | Coreset Patch Memory Bank | 0.3167 | 0.8486 | 0.6815 | 1.64 |
| EfficientAD | Student-Teacher Autoencoder | 0.4667 | 0.6606 | 0.5910 | 27.14 |
| DRAEM | Synthetic Reconstructive Net | 0.6250 | 0.6613 | 0.5840 | 5.08 |
| AnomalyCLIP | Object-Agnostic Prompt Tuning | 0.3333 | 0.1588 | 0.2210 | 5.04 |

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/vista-vlm/vista-vlm.git
cd vista-vlm
pip install -r requirements.txt
```

### 2. Validate Environment
Verify hardware capabilities, PyTorch configuration, and dependencies:
```bash
python scripts/validate_environment.py
```

### 3. Prepare Benchmark Datasets
Create standardized, ground-truth-validated benchmark splits:
```bash
python scripts/prepare_benchmark_data.py
```

### 4. Train Model & Calibrate Conformal Reliability
Train VISTA-VLM on normal samples and calibrate conformal thresholds:
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 2
```

### 5. Run Benchmark Baselines & Ablations
Reproduce genuine benchmark comparisons across all baseline methods:
```bash
python scripts/run_baselines.py --category bottle
python vista_vlm/experiments/run_ablation.py
```

### 6. Run Test Suite
Execute the comprehensive 31-test pytest validation suite:
```bash
python -m pytest tests/
```

### 7. Launch FastAPI Server & Visual Intelligence Studio
Start the backend API server:
```bash
uvicorn vista_vlm.api.app:app --host 0.0.0.0 --port 8000
```
In a separate terminal, launch the VISTA Studio dashboard:
```bash
cd studio
npm run build
npm run preview
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Interactive Studio Modules

1. **Visual Inspection**: One-click inspection with real-time status badges (`NORMAL`, `ANOMALOUS`, `UNCERTAIN`, `OOD`), linear/circular score gauges, and selective prediction review advisories.
2. **Spatial Visualization**: Side-by-side and layered anomaly map viewer with interactive bounding boxes, opacity sliders, and connected component metrics.
3. **Ask VISTA**: Grounded multimodal Q&A surfacing verified CLIP prompt rankings, similarity scores, ambiguity alerts, and inspector correction forms.
4. **Research Lab**: Checkpoint metadata inspector, architecture ablation comparisons (Global vs Local, Multi-Layer vs Single, ViT-S vs ViT-B), and few-shot learning curves ($k \in \{1, 2, 4, 8\}$).
5. **Benchmark Center**: Sortable baseline comparison table, ROC/PR curves, conformal calibration reliability diagrams, and CPU latency profiles.

---

## Repository Structure

```
VISTA-VLM/
├── benchmarks/              # Genuine benchmark JSON results, reports, feedback
├── checkpoints/             # Trained model weights and calibration files
├── config/                  # YAML configurations (default, mvtec, visa)
├── data/                    # Processed datasets, ground truth masks, synthetic samples
├── docs/                    # Technical research report, architecture, API guides
│   ├── RESEARCH_REPORT.md   # Comprehensive 19-section academic research report
│   ├── ARCHITECTURE.md      # Deep-dive system architecture specification
│   ├── DATASET_GUIDE.md     # Ingestion, normalization, and split protocol
│   ├── API_REFERENCE.md     # OpenAPI / REST endpoint documentation
│   └── MODEL_CARD.md        # Production model card and ethical considerations
├── scripts/                 # Training, benchmark, download, and testing scripts
├── studio/                  # VISTA Visual Intelligence Studio (React 19 / Vite / TS)
├── tests/                   # 31-unit/integration test suite (100% passing)
├── vista_vlm/               # Core Python package
│   ├── api/                 # FastAPI routes, schemas, and application
│   ├── calibration/         # Split conformal, OOD, and selective prediction
│   ├── config/              # Pydantic configuration schemas
│   ├── data/                # Dataset loaders (MVTec, VisA, BTAD, MPDD, Synthetic)
│   ├── models/              # DINOv2, CLIP, Fusion, Decoders, Baselines, Pipeline
│   ├── optimization/        # TorchScript exporter and benchmark profiler
│   └── utils/               # Logger, registry, metrics
├── pyproject.toml           # Project metadata & build tool definition
└── requirements.txt         # Pinned python dependencies
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
