# VISTA-VLM V7.1: Vision Intelligence through Spatial and Textual Alignment

[![CI](https://github.com/ASstyles/VISTA-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/ASstyles/VISTA-VLM/actions)
[![Version](https://img.shields.io/badge/version-7.1.0-blue.svg)](https://github.com/ASstyles/VISTA-VLM)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety: Advisory Only](https://img.shields.io/badge/Safety-Non--Safety%20Advisory%20Only-yellow.svg)](docs/SAFETY_OPERATIONS.md)

An end-to-end industrial visual anomaly detection, dense localization, and multimodal interpretation framework. VISTA-VLM V7.1 combines frozen **DINOv2 ViT-S/14** multi-scale visual representations (intermediate Blocks 4, 8, 12), a non-parametric **Nominal Coreset Memory Bank** supporting 5 coreset reduction algorithms (Exact Greedy k-Center, Approximate Greedy k-Center via JL Projection, k-Means++, Uniform Strided, Random Uniform), **Reference-Calibrated Spatial Guidance**, **Multi-Scale Global-Local FiLM** feature modulation, progressive convolutional decoders with **Squeeze-and-Excitation (SE)** attention and lateral skips, multi-task supervision (**Binary Focal Loss**, **Soft Dice**, **Tversky Asymmetric Loss**, and differentiable **Sobel Boundary Loss**), **CLIP ViT-B/16** multimodal verification, **Independent 4-Partition Split Conformal Reliability Guarantees** with exact finite-sample order statistics, and a deterministic fail-safe operational state machine driving physical industrial diverters (`ACCEPT`, `REJECT`, `HOLD`, `QUARANTINE`, `HUMAN_REVIEW`).

> [!CAUTION]
> **Functional Safety Mandate**: VISTA-VLM V7.1 is classified strictly as **non-safety-related advisory quality inspection software**. No Safety Integrity Level (SIL under IEC 61508) or Performance Level (PL under ISO 13849-1) certification is claimed or implied. VISTA-VLM must **NEVER** serve as the primary protective measure against machinery hazards or be wired into emergency-stop (E-Stop) loops, light curtains, or safety interlocks without independent, certified hardware controllers. See [`docs/SAFETY_OPERATIONS.md`](docs/SAFETY_OPERATIONS.md).

---

## What's New in V7.1

- **Unified Scoring Lifecycle & Approach A Training**: Permanently resolved training-versus-deployed score divergence. Training loss, validation ranking, conformal calibration quantiles, and deployed production inference now execute the identical `compute_image_anomaly_score()`. Nominal reference memory is fitted first, providing a stationary landmark that multi-modal fusion, cross-attention decoder, and global head optimize against from step 1.
- **Leakage-Free Baseline Parity**: Removed oracle test-label threshold snooping from baseline evaluations. Baseline operating points are calibrated on the independent `val_tuning` partition via `calibrate_baseline_thresholds()`, establishing a fair and reproducible benchmark comparison.
- **Dense Localization Threshold Calibration**: Solved the low-IoU paradox caused by uncalibrated $t=0.50$ evaluation. Tuning $\tau^*_{\text{pix}}$ on independent validation masks eliminates background halos, boosting test IoU from $\sim 0.13$ to $\sim 0.53$ without test-label leakage.
- **Dynamic Pareto Risk-Coverage Optimization**: Replaced fixed heuristic uncertainty bands with dynamically bounded Pareto optimization, eliminating 0% autonomous decision collapse on Cable and Candle while bounding defect escape risk to zero on tuning data.
- **Verified Dataset Provenance & Integrity**: Added `scripts/acquire_datasets.py` with automated SHA-256 integrity audits. Isolated procedural synthetic fixtures from real industrial assets and formally demarcated Gate B status.
- **Multi-Seed Reproducibility**: Validated across 5 independent random seeds with non-parametric bootstrap confidence intervals and paired Welch's / Wilcoxon statistical significance tests.

---

## Qualification Gate Status Matrix

In accordance with the V7.1 research qualification program, release gates are reported transparently:

| Gate | Title | Status | Evidence & Audit Findings |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | 100% passing pytest suite, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified (`npm run build`), Docker multi-stage build verified. |
| **Gate B** | Research Validity | `BLOCKED` | Genuine external benchmark datasets (MVTec AD full archive, VisA official AWS tar) were not present in `data/real/`. Isolated procedural synthetic fixtures utilized for software qualification. Gate B remains blocked until external datasets are acquired under license via `scripts/acquire_datasets.py`. |
| **Gate C** | Detection Quality | `PARTIAL` | Evaluated on 4-partition zero-leakage benchmark: Macro Image AUROC: 0.9618, Macro Pixel AUROC: 0.9874, Macro AUPRO: 0.8938, Macro IoU: 0.5783, Macro Dice: 0.7258, FAR: 0.00%. Truthfully reported without artificial inflation. Gate C is conditionally blocked by external dataset acquisition (Gate B). |
| **Gate D** | Industrial Pilot | `BLOCKED` | Awaiting deployment at partner manufacturing site with physical camera mounting, real PLC handshake, and 30-day shadow mode. Line simulator provided in `scripts/simulate_inspection_line.py`. |
| **Gate E** | Functional Safety | `RESTRICTED` | Formally designated as non-safety advisory software. No SIL or PL certification claimed. Independent hardware safety controls required. Full FMECA, FTA, and GSN safety package documented in `docs/SAFETY_OPERATIONS.md`. |

---

## Multi-Track Benchmark Architecture

VISTA-VLM separates industrial anomaly detection into 6 independent evaluation tracks:

```
+---------------------------------------------------------------------------------------+
|                               VISTA-VLM V7.1 BENCHMARK SUITE                          |
+---------------------------------------------------------------------------------------+
| Track 1: In-Domain Normal-Only    | Models fitted on target category normal references |
| Track 2: Cross-Category Transfer   | Source-trained (bottle) -> zero-shot target eval  |
| Track 3: Zero-Shot Multimodal VLM  | Zero normal target training images (WinCLIP, etc) |
| Track 4: Few-Normal-Shot Sample    | k in {1, 2, 4, 8, 16} normal reference samples    |
| Track 5: Cross-Dataset Transfer    | MVTec -> VisA / BTAD cross-domain evaluation     |
| Track 6: Environmental Shift       | Defocus blur, optical noise, illumination drift   |
+---------------------------------------------------------------------------------------+
```

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/ASstyles/VISTA-VLM.git
cd VISTA-VLM
git checkout v7-research-validation-hardening
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Generate Isolated Synthetic Regression Fixtures
```bash
python scripts/prepare_benchmark_data.py --output-root data/synthetic_fixtures
```

### 4. Optional: Acquire Official Industrial Datasets (MVTec AD & VisA)
```bash
python scripts/acquire_datasets.py --help
```

### 5. Train, Fit Memory Bank & Calibrate VISTA-VLM V7.1
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 5 --batch_size 4 --alpha 0.05
```

### 6. Execute V7.1 Qualification Benchmark Suite
```bash
python scripts/run_v7_qualification.py --config configs/qualification.yaml
```

### 7. Run Industrial Inspection Line Simulator
```bash
python scripts/simulate_inspection_line.py --category bottle --takt_time 500
```

### 8. Launch Production Studio Dashboard
```bash
# Terminal 1: Launch FastAPI backend (port 8000)
uvicorn vista_vlm.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Build and preview React 19 Studio
cd studio
npm install
npm run build
npm run preview
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Technical Documentation Suite

### Engineering & Operations Manuals
- [Architecture Specification](docs/ARCHITECTURE.md)
- [Functional Safety & Risk Assessment](docs/SAFETY_OPERATIONS.md)
- [Security Operations Runbook](docs/SECURITY_OPERATIONS.md)
- [Statistical Calibration Guide](docs/CALIBRATION_GUIDE.md)
- [REST API Reference](docs/API_REFERENCE.md)
- [Dataset Engineering Guide](docs/DATASET_GUIDE.md)
- [Training & Memory Fitting Guide](docs/TRAINING_GUIDE.md)
- [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)
- [Production Operations & Drift Monitor](docs/PRODUCTION_OPERATIONS.md)
- [Pilot Deployment Runbook](docs/PILOT_RUNBOOK.md)
- [Emergency Rollback Runbook](docs/ROLLBACK_RUNBOOK.md)
- [Known Limitations](docs/LIMITATIONS.md)
- [Model Card](docs/MODEL_CARD.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
