# VISTA-VLM V6.0: Vision Intelligence through Spatial and Textual Alignment

[![CI](https://github.com/ASstyles/VISTA-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/ASstyles/VISTA-VLM/actions)
[![Version](https://img.shields.io/badge/version-6.0.0-blue.svg)](https://github.com/ASstyles/VISTA-VLM)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-95%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety: Advisory Only](https://img.shields.io/badge/Safety-Non--Safety%20Advisory%20Only-yellow.svg)](docs/SAFETY_OPERATIONS.md)

An end-to-end industrial visual anomaly detection, dense localization, and multimodal interpretation framework. VISTA-VLM combines frozen **DINOv2 ViT-S/14** multi-scale visual representations (intermediate Blocks 4, 8, 12), a non-parametric **Nominal Coreset Memory Bank** with Exact & Approximate Greedy k-Center sampling, **Reference-Calibrated Spatial Guidance**, **Multi-Scale Global-Local FiLM** feature modulation, progressive convolutional decoders with **Squeeze-and-Excitation (SE)** attention and lateral skips, multi-task supervision (**Binary Focal Loss**, **Soft Dice**, **Tversky Asymmetric Loss**, and differentiable **Sobel Boundary Loss**), **CLIP ViT-B/16** multimodal verification, **Split Conformal Reliability Guarantees** with exact finite-sample order statistics, and a deterministic 9-state fail-safe operational state machine driving physical industrial diverters (`ACCEPT`, `REJECT`, `HOLD`, `QUARANTINE`, `HUMAN_REVIEW`).

> [!CAUTION]
> **Functional Safety Mandate**: VISTA-VLM V6.0 is classified strictly as **non-safety-related advisory quality inspection software**. No Safety Integrity Level (SIL under IEC 61508) or Performance Level (PL under ISO 13849-1) certification is claimed or implied. VISTA-VLM must **NEVER** serve as the primary protective measure against machinery hazards or be wired into emergency-stop (E-Stop) loops, light curtains, or safety interlocks without independent, certified hardware controllers. See [`docs/SAFETY_OPERATIONS.md`](docs/SAFETY_OPERATIONS.md).

---

## What's New in V6.0

- **High-Discrimination Image-Level Anomaly Detection**: Resolved V5 image-level scoring degradation by eliminating destructive per-image min-max distortion. Established category-specific reference normal distribution statistics ($\mu, \sigma, p_{95}, \text{scale\_range}$) derived strictly from training data to preserve cross-image absolute distance ranking. Introduced a hybrid scoring head combining calibrated memory distance ($70\%$) with spatial top-$k$ pooling ($30\%$).
- **Exact & Approximate Greedy k-Center Coreset Selection**: Implemented minimax manifold coverage via exact greedy k-center ($c_m = \arg\max_i \min_{c \in C} \|x_i - c\|_2$) and Johnson-Lindenstrauss random orthogonal projection ($D \to 64$), accelerating pairwise distance computations by $8\times$ while preserving geometric manifold bounds.
- **High-Resolution Defect Localization & Crack Detection**: Combined multi-scale skips (DINOv2 Block 4 fine details + Block 8 semantics) with asymmetric Tversky loss ($\alpha=0.7, \beta=0.3$) and differentiable Sobel boundary gradient loss to suppress blurry heatmaps and achieve sharp boundary contours on micro-cracks ($<0.1\%$ image area).
- **Production-Grade Checkpoint & Memory Bank Lifecycle**: Unified neural network weights, nominal memory bank patch vectors, category reference statistics, and training metadata into single versioned artifacts verified by SHA-256 digests. Implemented explicit operational states: `UNINITIALIZED`, `LOADING`, `READY`, `INVALID`, `UNAVAILABLE`.
- **Statistically Defensible Conformal Calibration**: Implemented finite-sample split conformal rank calculation $k = \lceil(n+1)(1-\alpha)\rceil$ with strict rejection ($k > n \implies \text{InsufficientSampleSizeError}$). Aligned anomaly decisions with upper-tail conformal p-values ($p(x) = \frac{1 + \sum_{i} \mathbb{I}(s_i \ge s_{\text{test}})}{n+1}$).
- **Regularized Mahalanobis OOD Detection**: Integrated Ledoit-Wolf shrinkage covariance estimation with matrix conditioning checks, preventing singular matrix inversions and enforcing fail-closed escalation (`QUARANTINE`) on uncalibrated query distributions.
- **Selective Prediction Tuning on Labelled Data**: Permanently eliminated the V5 normal-only label bug (`val_labels = [0] * len(cal_scores)`). Calibrated the empirical uncertainty band $\delta$ on a properly separated validation set containing both normal and synthetic defect instances, evaluated with Wilson score intervals and Clopper-Pearson exact binomial bounds.
- **Cybersecurity & Container Hardening**:
  - `SEC-V6-01`: Path traversal eliminated in `POST /inspect/batch` via canonical `realpath` containment sandboxing (`validate_safe_path`).
  - `SEC-V6-02`: Streaming chunked ingestion with 20MB cap (`MAX_IMAGE_BYTES`) eliminates memory exhaustion DoS.
  - `SEC-V6-03`: Decompression bomb protection via 25M pixel limit, magic byte validation, and $[16, 4096]$ dimension limits.
  - `SEC-V6-04`: Multi-stage Docker container executing under an unprivileged non-root user (`vista:vista`, UID 10001).
- **Industrial Inspection Line Simulator**: Built `scripts/simulate_inspection_line.py` modeling an end-to-end industrial cell: GigE camera acquisition, optical quality checks (Laplacian blur variance, illumination lux limits), VISTA V6 inference, PLC diverter control, tamper-evident audit logging, and takt-time latency accounting.
- **Frontend Industrial Inspection Studio**: Upgraded React 19 + TypeScript + Vite dashboard with real-time inspection feeds, spatial heatmaps, bounding boxes, defect region classification, selective decision banners, and system health status.
- **Comprehensive Quality Engineering**: Expanded automated test suite to **95 unit, security, and integration tests (100% passing)** across Python 3.10, 3.11, and 3.12 CI matrix with Ruff linting and zero warnings.

---

## Qualification Gate Status Matrix

In accordance with Phase 13 of the V6 qualification program, release gates are reported transparently:

| Gate | Title | Status | Evidence & Audit Findings |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | 95 passing pytest tests, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified, Docker multi-stage build verified. |
| **Gate B** | Research Validity | `BLOCKED` | Genuine external benchmark datasets (MVTec AD full archive, VisA official AWS tar) were not present in `data/real/`. Isolated procedural synthetic fixtures utilized for software qualification. Gate B remains blocked until external datasets are acquired under license. |
| **Gate C** | Detection Quality | `PASS` | Evaluated on isolated fixtures: Macro Image AUROC $\ge 0.90$, Pixel AUROC $\ge 0.95$, AUPRO $\ge 0.80$, false-alarm operating point $\le 5\%$, tiny defect recall sustained. |
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
