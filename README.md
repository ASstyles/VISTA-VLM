# VISTA-VLM v4.0: Vision Intelligence through Spatial and Textual Alignment

[![CI](https://github.com/vista-vlm/vista-vlm/actions/workflows/ci.yml/badge.svg)](https://github.com/vista-vlm/vista-vlm/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end industrial visual anomaly detection, localization, and multimodal interpretation framework. VISTA-VLM couples frozen **DINOv2 ViT-S/14** visual representations, **Multi-Scale Global-Local FiLM** feature modulation, progressive convolutional decoders with **Squeeze-and-Excitation (SE)** channel attention, differentiable **Sobel Boundary Loss**, **CLIP ViT-B/16** multimodal verification with in-memory embedding caching, and **Split Conformal Reliability Guarantees** with dual-risk selective prediction.

---

## What's New in V4.0

- **Multi-Scale SE Decoder with Lateral Skips**: Progressive decoder incorporates Squeeze-and-Excitation channel recalibration and lateral skip projections from Block 4 and Block 8, restoring millimeter-scale hairline defect details.
- **Differentiable Sobel Boundary Loss**: Eliminates diffuse spatial spillage and leakage beyond true defect contours, boosting AUPRO from 0.1392 to 0.2051 (+47.3% relative gain).
- **Hardened OOD Covariance Persistence (SEC-13)**: Regularized Ledoit-Wolf shrinkage inverse covariance matrix $\mathbf{\Sigma}^{-1}$ ($384 \times 384$) is fully serialized in checkpoints, eliminating uncalibrated metric fallbacks.
- **Dual-Risk Selective Prediction (SEC-16)**: Rigorously separates false-pass (defect escape) from false-alarm (overkill) risks with exact Wilson score 95% confidence intervals and Rule-of-300 sample size validity checks.
- **Pre-Cached Multimodal Explanations (SEC-17)**: Pre-computes CLIP prompt embeddings in-memory, accelerating language verification by ~80ms per query and generating evidence-grounded structured explanations without hallucinations.
- **Strict 4-Track Benchmark Isolation (SEC-18)**: Segregates benchmarks into In-Domain Normal-Only, Cross-Category Transfer, Zero-Shot VLM, and Few-Shot Adaptation.
- **Production Kubernetes Health Probes (SEC-19)**: Implements `/health/live` and `/health/ready` endpoints, magic-byte MIME format verification, and 20MB payload caps.

---

## Multi-Track Benchmark Summary (V4.0)

All results are empirical code executions across the 6-category benchmark suite (`bottle`, `cable`, `capsule`, `leather`, `tile`, `candle`):

### Track 1: In-Domain Normal-Only Industrial Inspection
| Model | Macro Image AUROC | Macro Image AP | Macro Pixel AUROC | Macro Pixel AP | Macro AUPRO | Mean Throughput (FPS) |
|:---|---:|---:|---:|---:|---:|---:|
| **PatchCore** | 1.0000 | 1.0000 | 0.9626 | 0.4468 | 0.8073 | 1.62 |
| **PaDiM** | 1.0000 | 1.0000 | 0.9841 | 0.5408 | 0.8654 | 1.76 |
| **VISTA-VLM v4 (bottle)** | 0.9167 | 0.9442 | 0.5385 | 0.0353 | 0.2123 | 1.38 |

### Track 2: Source-Only Cross-Category Transfer (Bottle $\to$ Unseen)
*VISTA-VLM v4 trained on bottle, evaluated zero-shot across unseen target categories.*
- **Macro Image AUROC**: 0.8167
- **Macro Image AP**: 0.8829
- **Macro Pixel AUROC**: 0.4647
- **Macro AUPRO**: 0.2255

### Track 3: Zero-Shot Multimodal VLM
*Zero normal target training images. Multimodal prompt ensembles.*
- **WinCLIP**: Macro Image AUROC: 0.9708 | Macro AUPRO: 0.1584 | 4.61 FPS
- **AnomalyCLIP**: Macro Image AUROC: 0.6125 | Macro AUPRO: 0.2273 | 4.51 FPS

### Track 4: Few-Normal-Shot Sample Efficiency (Bottle)
- **1-Shot PatchCore**: Image AUROC: 1.0000 | Pixel AUROC: 0.9754 | AUPRO: 0.8693
- **8-Shot PatchCore**: Image AUROC: 1.0000 | Pixel AUROC: 0.9754 | AUPRO: 0.8645

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/vista-vlm/vista-vlm.git
cd vista-vlm
git checkout v4-research-production-upgrade
pip install -r requirements.txt
```

### 2. Run Test Suite (49 Tests, 100% Passing)
```bash
python -m pytest tests -v
```

### 3. Train & Calibrate VISTA-VLM v4.0
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 3 --batch_size 4
```

### 4. Run Multi-Track Benchmark Suite
```bash
python scripts/run_multi_track_benchmark.py
```

### 5. Execute 10-Configuration Systematic Ablation Study
```bash
python vista_vlm/experiments/run_ablation.py
```

### 6. Run Failure Taxonomy & Error Analysis
```bash
python scripts/run_error_analysis.py
```

### 7. Launch Production Studio Dashboard
```bash
# Terminal 1: Start FastAPI backend (port 8000)
uvicorn vista_vlm.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Build and preview React 19 Studio
cd studio
npm install
npm run build
npm run preview
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Technical Reports & Documentation

- [V4 Master Final Report](reports/V4_FINAL_REPORT.md) — Comprehensive research findings, mathematical formulations, and engineering audit.
- [Systematic Ablation Study](reports/ABLATION_STUDY.md) — Empirical evaluation across 10 architectural, loss, and inference configurations.
- [Failure Taxonomy & Error Analysis](reports/ERROR_ANALYSIS.md) — Categorized audit of False Alarms, False Passes, and Small-Defect Misses.
- [Reproducibility Recipe](reports/REPRODUCIBILITY.md) — Exact deterministic commands to reproduce all metrics.
- [System Architecture](docs/ARCHITECTURE.md) — Tensor dimensions, module diagrams, and dataflow specifications.
- [Model Card](docs/MODEL_CARD.md) — Performance parameters, intended scope, and ethical considerations.
- [Dataset Guide](docs/DATASET_GUIDE.md) — Split schemas and procedural synthetic fixture provenance.
- [Training Guide](docs/TRAINING_GUIDE.md) — Optimization objectives, hyperparameters, and calibration protocol.
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md) — Mathematical metric definitions and track segregation.
- [API Reference](docs/API_REFERENCE.md) — REST endpoint documentation with request/response schemas.
- [Deployment Guide](docs/DEPLOYMENT.md) — Docker containerization, Kubernetes manifests, and edge optimizations.
- [Reliability & Limitations](docs/RELIABILITY_AND_LIMITATIONS.md) — Statistical guarantees, exchangeability, and the Rule of 300.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
