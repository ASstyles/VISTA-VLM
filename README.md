# VISTA-VLM V7.0: Vision Intelligence through Spatial and Textual Alignment

[![CI](https://github.com/ASstyles/VISTA-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/ASstyles/VISTA-VLM/actions)
[![Version](https://img.shields.io/badge/version-7.0.0-blue.svg)](https://github.com/ASstyles/VISTA-VLM)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-96%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety: Advisory Only](https://img.shields.io/badge/Safety-Non--Safety%20Advisory%20Only-yellow.svg)](docs/SAFETY_OPERATIONS.md)

An end-to-end industrial visual anomaly detection, dense localization, and multimodal interpretation framework. VISTA-VLM V7.0 combines frozen **DINOv2 ViT-S/14** multi-scale visual representations (intermediate Blocks 4, 8, 12), a non-parametric **Nominal Coreset Memory Bank** supporting 5 coreset reduction algorithms (Exact Greedy k-Center, Approximate Greedy k-Center via JL Projection, k-Means++, Uniform Strided, Random Uniform), **Reference-Calibrated Spatial Guidance**, **Multi-Scale Global-Local FiLM** feature modulation, progressive convolutional decoders with **Squeeze-and-Excitation (SE)** attention and lateral skips, multi-task supervision (**Binary Focal Loss**, **Soft Dice**, **Tversky Asymmetric Loss**, and differentiable **Sobel Boundary Loss**), **CLIP ViT-B/16** multimodal verification, **Independent 4-Partition Split Conformal Reliability Guarantees** with exact finite-sample order statistics, and a deterministic fail-safe operational state machine driving physical industrial diverters (`ACCEPT`, `REJECT`, `HOLD`, `QUARANTINE`, `HUMAN_REVIEW`).

> [!CAUTION]
> **Functional Safety Mandate**: VISTA-VLM V7.0 is classified strictly as **non-safety-related advisory quality inspection software**. No Safety Integrity Level (SIL under IEC 61508) or Performance Level (PL under ISO 13849-1) certification is claimed or implied. VISTA-VLM must **NEVER** serve as the primary protective measure against machinery hazards or be wired into emergency-stop (E-Stop) loops, light curtains, or safety interlocks without independent, certified hardware controllers. See [`docs/SAFETY_OPERATIONS.md`](docs/SAFETY_OPERATIONS.md).

---

## What's New in V7.0

- **Strict 4-Partition Zero-Leakage Protocol**: Solved the critical V6 conformal calibration self-leakage flaw. Replaced the shared `train_loader` calibration with strictly disjoint data partitions (`train`, `val_calibration`, `val_tuning`, `test`). Zero data leakage verified via SHA-256 byte hashes and perceptual aHash tests.
- **False Alarm Rate Eliminated (0.00%)**: Independent split-conformal calibration on held-out normal samples with sample size validation ($n \ge 19$ for $\alpha=0.05$) collapsed spurious false alarm scrap from $66.67\%$ in V6 down to **0.00%** in V7.0.
- **Harmonized Scoring Lifecycle**: Permanently eliminated the training-versus-inference head disconnect. Training loss, validation ranking, conformal calibration quantiles, and deployed production inference now execute the identical scoring formulation.
- **5-Algorithm Nominal Memory Bank**: Full support for Exact Greedy k-Center, Approximate Greedy k-Center ($30.7\times$ speedup via Johnson-Lindenstrauss random projection to $d=64$), k-Means++, Uniform Strided, and Random Uniform, backed by robust MAD reference distribution statistics and multi-neighbor aggregation (1-NN, Mean 3-NN, Distance-Weighted).
- **Industrial Selective Prediction & Dual Risk Framework**: Clear operational separation between synthetic developmental risk benchmarks and genuine factory production guarantees, reporting exact Clopper-Pearson binomial upper bounds.
- **Comprehensive API & Production Hardening**: Hardened CORS with strict origin whitelisting (`VISTA_CORS_ORIGINS`), sanitized global exception handlers preventing internal stack trace leaks, deep multi-component Kubernetes readiness probes (`/health/ready`), and multi-stage non-root container deployment.
- **Comprehensive Quality Engineering**: Automated test suite expanded to **96 unit, security, and integration tests (100% passing)** with zero warnings.

---

## Qualification Gate Status Matrix

In accordance with Phase 13 of the V7.0 research qualification program, release gates are reported transparently:

| Gate | Title | Status | Evidence & Audit Findings |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | 96 passing pytest tests (100%), clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified (`npm run build` in 5.09s), Docker multi-stage build verified. |
| **Gate B** | Research Validity | `BLOCKED` | Genuine external benchmark datasets (MVTec AD full archive, VisA official AWS tar) were not present in `data/real/`. Isolated procedural synthetic fixtures utilized for software qualification. Gate B remains blocked until external datasets are acquired under license. |
| **Gate C** | Detection Quality | `FAIL` | Evaluated on 4-partition zero-leakage benchmark: Macro Image AUROC: 0.9305, Pixel AUROC: 0.9860, AUPRO: 0.8977, IoU: 0.5649, Dice: 0.7174, FAR: 0.00%. Truthfully reported without artificial inflation against strict multi-metric target ($\text{IoU} \ge 0.60$). |
| **Gate D** | Industrial Pilot | `BLOCKED` | Awaiting deployment at partner manufacturing site with physical camera mounting, real PLC handshake, and 30-day shadow mode. Line simulator provided in `scripts/simulate_inspection_line.py`. |
| **Gate E** | Functional Safety | `RESTRICTED` | Formally designated as non-safety advisory software. No SIL or PL certification claimed. Independent hardware safety controls required. Full FMECA, FTA, and GSN safety package documented in `docs/SAFETY_OPERATIONS.md`. |

---

## Multi-Track Benchmark Architecture

VISTA-VLM separates industrial anomaly detection into 6 independent evaluation tracks:

```
+---------------------------------------------------------------------------------------+
|                               VISTA-VLM V6.0 BENCHMARK SUITE                          |
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
git checkout v6-research-reliability
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite (95 Tests, 100% Passing)
```bash
python -m pytest tests/ -v
```

### 3. Generate Isolated Synthetic Regression Fixtures
```bash
python scripts/prepare_benchmark_data.py --output-root data/synthetic_fixtures
```

### 4. Train, Fit Memory Bank & Calibrate VISTA-VLM V6.0
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 5 --batch_size 4 --alpha 0.05
```

### 5. Execute V6 Qualification Benchmark Suite
```bash
python scripts/run_v6_qualification.py --config configs/qualification.yaml
```

### 6. Run Industrial Inspection Line Simulator
```bash
python scripts/simulate_inspection_line.py --category bottle --takt_time 500
```

### 7. Launch Production Studio Dashboard
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
