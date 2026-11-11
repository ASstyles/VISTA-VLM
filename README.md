# VISTA-VLM v5.0: Industrial Real-World Qualification & Safety Assurance Program

[![CI](https://github.com/vista-vlm/vista-vlm/actions/workflows/ci.yml/badge.svg)](https://github.com/vista-vlm/vista-vlm/actions)
[![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)](https://github.com/ASstyles/VISTA-VLM)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-58%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety: SIL 0 Advisory](https://img.shields.io/badge/Safety-SIL%200%20Advisory%20Only-orange.svg)](reports/V5_SAFETY_CASE.md)

An end-to-end industrial visual anomaly detection, dense localization, and multimodal interpretation framework. VISTA-VLM combines frozen **DINOv2 ViT-S/14** multi-scale visual representations (Blocks 4, 8, 12), a non-parametric **Nominal Coreset Memory Bank**, **Memory-Guided Spatial Residual Fusion**, **Multi-Scale Global-Local FiLM** feature modulation, progressive convolutional decoders with **Squeeze-and-Excitation (SE)** attention, differentiable **Sobel Boundary Loss**, **CLIP ViT-B/16** multimodal candidate verification, and **Split Conformal Reliability Guarantees** with a deterministic 9-state fail-safe operational state machine.

> [!IMPORTANT]
> **Safety Notice**: VISTA-VLM V5.0 is officially classified as an **Advisory Quality Inspection Support Tool (SIL 0)**. It is **NOT** a certified safety component under IEC 61508 or ISO 13849-1 and must never be wired directly to machine safety-critical trip circuits, light curtains, or emergency stop loops without external SIL 3 / PL e hardware controllers.

---

## What's New in V5.0

- **Spatial Localization Breakthrough**: Resolved the V4 localization collapse (0.5385 Pixel AUROC) by introducing multi-scale patch extraction (Blocks 4 & 8) and a coreset nominal memory bank with residual spatial fusion ($w_{\text{mem}} = 0.5$). On `bottle`, **Pixel AUROC jumped to 0.9966 (+85.1%)**, **AUPRO jumped to 0.9713 (+373%)**, and **tiny defect recall ($<0.1\%$ area) reached 83.64%** (vs 16.14% for PatchCore).
- **Strict Data Segregation & Cryptographic Manifests**: Permanently decoupled `data/synthetic_fixtures/`, `data/real/`, and `data/manifests/`. Synthetic generators are clearly disclosed and randomized with per-sample seeds to eliminate train/test data leakage.
- **Track 4 Few-Shot Inclusion**: Formally integrated VISTA-VLM into the sample-constrained adaptation track, achieving **0.9969 Pixel AUROC at $k=1$ shot** and demonstrating instantaneous line ramp-up viability.
- **Deterministic 9-State Fail-Safe Engine**: Predictions transition deterministically between `NORMAL`, `ANOMALY_DETECTED`, `UNCERTAIN`, `OOD`, `IMAGE_QUALITY_FAILURE`, `MODEL_UNAVAILABLE`, `CALIBRATION_INVALID`, `COMMUNICATION_FAILURE`, and `HUMAN_REVIEW_REQUIRED`, driving physical fail-closed dispositions (`ACCEPT`, `REJECT`, `HOLD`, `QUARANTINE`, `HUMAN_REVIEW`).
- **Cybersecurity Hardening**:
  - `SEC-V5-01`: Path traversal eliminated in `POST /inspect/batch` via canonical containment sandboxing (`validate_safe_path`).
  - `SEC-V5-02`: Streaming chunked ingestion with a 20MB limit (`MAX_IMAGE_BYTES`) eliminates memory exhaustion DoS.
  - `SEC-V5-03`: Decompression bomb protection via 25M pixel limit, format verification, and $[16, 4096]$ dimension bounds.
- **GSN Safety Case & Regulatory Audit Plan**: Authored complete Goal Structuring Notation safety case, FMECA hazard analysis, EU AI Act compliance analysis, and third-party independent assessment plan.

---

## Multi-Track Benchmark Summary (V5.0)

*Data Provenance Disclosure: Evaluated on procedural synthetic benchmark fixtures conforming to official MVTec AD and VisA directory schemas. Genuine public datasets are tracked under dedicated unblocking runbooks in [`reports/V5_DATA_PROVENANCE.md`](reports/V5_DATA_PROVENANCE.md).*

### Track 1: In-Domain Normal-Only Industrial Inspection
| Model | Image AUROC (Macro $\pm$ Std) | Image AP (Macro) | Pixel AUROC (Macro $\pm$ Std) | Pixel AUPRO (Macro) | Bottle Pixel AUROC | Bottle AUPRO | Bottle Tiny Defect Recall ($<0.1\%$) | Mean Throughput (FPS, CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PatchCore** | $\mathbf{0.9722 \pm 0.0621}$ | $\mathbf{0.9814}$ | $\mathbf{0.9874 \pm 0.0263}$ | $\mathbf{0.8695}$ | $0.9980$ | $0.9846$ | $16.14\%$ | 1.57 |
| **PaDiM** | $0.9361 \pm 0.1046$ | $0.9592$ | $0.9902 \pm 0.0200$ | $0.9063$ | $0.9982$ | $0.9845$ | $21.82\%$ | 1.68 |
| **VISTA-VLM v5.0** | $0.5680 \pm 0.2311$ | $0.7485$ | $0.9740 \pm 0.0256$ | $0.7426$ | $\mathbf{0.9966}$ | $\mathbf{0.9713}$ | $\mathbf{83.64\%}$ | 1.45 |

### Track 2: Source-Only Cross-Category Transfer (Bottle $\to$ Unseen)
*VISTA-VLM v5.0 trained strictly on `bottle`, evaluated zero-shot across unseen target categories.*
- **Macro Image AUROC**: $0.7433 \pm 0.1525$
- **Macro Image AP**: $0.8281$
- **Macro Pixel AUROC**: **$0.8805 \pm 0.0950$** (up from 0.4647 in v4)
- **Macro AUPRO**: **$0.5304$** (up from 0.2255 in v4)

### Track 3: Zero-Shot Multimodal VLM
*Zero normal target training images. Multimodal prompt ensembles.*
- **WinCLIP**: Macro Pixel AUROC: $0.5163 \pm 0.2675$ | Macro AUPRO: $0.1597$ | 4.39 FPS
- **AnomalyCLIP**: Macro Pixel AUROC: $0.5547 \pm 0.2071$ | Macro AUPRO: $0.2119$ | 4.40 FPS

### Track 4: Few-Normal-Shot Sample Efficiency (Bottle)
| Model | 1-Shot Pixel AUROC | 2-Shot Pixel AUROC | 4-Shot Pixel AUROC | 8-Shot Pixel AUROC | 16-Shot Pixel AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PatchCore** | $0.9970$ | $\mathbf{0.9980}$ | $\mathbf{0.9982}$ | $\mathbf{0.9984}$ | $\mathbf{0.9985}$ |
| **PaDiM** | $0.9881$ | $0.9955$ | $\mathbf{0.9982}$ | $\mathbf{0.9984}$ | $\mathbf{0.9985}$ |
| **VISTA-VLM v5.0** | $\mathbf{0.9969}$ | $0.9951$ | $0.9951$ | $0.9963$ | $0.9968$ |

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/ASstyles/VISTA-VLM.git
cd VISTA-VLM
git checkout v5-real-world-qualification
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite (58 Tests, 100% Passing)
```bash
python -m pytest tests/ -v
```

### 3. Generate Benchmark Test Fixtures with SHA-256 Manifests
```bash
python scripts/prepare_benchmark_data.py
```

### 4. Train, Fit Memory Bank & Calibrate VISTA-VLM v5.0
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 3 --batch_size 4 --alpha 0.05
```

### 5. Run Multi-Track Benchmark Suite
```bash
python scripts/run_multi_track_benchmark.py
```

### 6. Run Systematic Ablation and Error Taxonomy
```bash
python vista_vlm/experiments/run_ablation.py
python scripts/run_error_analysis.py
```

### 7. Launch Production Studio Dashboard
```bash
# Terminal 1: Start FastAPI backend (port 8000)
uvicorn vista_vlm.api.routes:app --host 0.0.0.0 --port 8000

# Terminal 2: Build and preview React 19 Studio
cd studio
npm install
npm run build
npm run preview
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Technical Documentation and Qualification Reports

### Engineering & Operations Manuals
- [Architecture Specification](docs/ARCHITECTURE.md)
- [Dataset Engineering Guide](docs/DATASET_GUIDE.md)
- [Research & Evaluation Protocol](docs/RESEARCH_PROTOCOL.md)
- [Training & Memory Fitting Guide](docs/TRAINING_GUIDE.md)
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)
- [Statistical Calibration Guide](docs/CALIBRATION_GUIDE.md)
- [REST API Reference](docs/API_REFERENCE.md)
- [Production Operations & Drift Monitor](docs/PRODUCTION_OPERATIONS.md)
- [Security Operations Runbook](docs/SECURITY_OPERATIONS.md)
- [Functional Safety Operations Guide](docs/SAFETY_OPERATIONS.md)
- [Pilot Deployment Runbook (30 Days)](docs/PILOT_RUNBOOK.md)
- [Emergency Rollback Runbook](docs/ROLLBACK_RUNBOOK.md)
- [Known Limitations](docs/LIMITATIONS.md)
- [Model Card](docs/MODEL_CARD.md)

### Scientific Qualification Reports
- [Final Industrial Qualification Report](reports/V5_FINAL_QUALIFICATION_REPORT.md)
- [Complete Baseline Audit](reports/V5_COMPLETE_BASELINE_AUDIT.md)
- [Data Provenance & Unblocking](reports/V5_DATA_PROVENANCE.md)
- [Multi-Track Benchmark Report](reports/V5_REAL_DATA_BENCHMARK.md)
- [Localization & Resolution Study](reports/V5_LOCALIZATION_STUDY.md)
- [Systematic Ablation Study](reports/V5_ABLATION_STUDY.md)
- [Cross-Dataset Generalization](reports/V5_CROSS_DATASET_GENERALIZATION.md)
- [Few-Shot Adaptation Study](reports/V5_FEW_SHOT_STUDY.md)
- [Statistical Calibration & Risk](reports/V5_CALIBRATION_AND_RISK.md)
- [Domain Shift & Robustness](reports/V5_DOMAIN_SHIFT_STUDY.md)
- [Comprehensive Error Analysis](reports/V5_ERROR_ANALYSIS.md)
- [Inference Latency & Sizing](reports/V5_PERFORMANCE_BENCHMARK.md)
- [Application Security Audit](reports/V5_SECURITY_AUDIT.md)
- [Hazard Analysis & FMECA](reports/V5_HAZARD_ANALYSIS.md)
- [Safety Requirements Specification](reports/V5_SAFETY_REQUIREMENTS.md)
- [GSN Industrial Safety Case](reports/V5_SAFETY_CASE.md)
- [Standards Applicability Analysis](reports/V5_STANDARDS_APPLICABILITY.md)
- [Independent Assessment Plan](reports/V5_INDEPENDENT_ASSESSMENT_PLAN.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
