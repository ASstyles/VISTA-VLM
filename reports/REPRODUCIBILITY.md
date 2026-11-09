# VISTA-VLM v4.0 — Complete Reproducibility Protocol & Recipe

## 1. System Environment & Prerequisites

### 1.1 Compute Specification
- **Operating System**: Windows 11 / Linux (Ubuntu 22.04 LTS recommended)
- **Python Version**: Python 3.10+ (Validated on Python 3.14.3)
- **PyTorch**: 2.6.0+ (CPU or CUDA 12.x compatible)
- **Node.js**: Node 18+ & npm 9+ (for Vite frontend)
- **Hardware Profile**:
  - Minimum: 4-Core CPU, 8 GB RAM, 10 GB disk space
  - Recommended: NVIDIA GPU (RTX 3080/4090 or A100), 16 GB VRAM

---

## 2. Environment Setup

### 2.1 Clone Repository and Checkout Upgrade Branch
```bash
git checkout v4-research-production-upgrade
```

### 2.2 Install Dependencies
```bash
# Python backend dependencies
pip install -e .
pip install pytest pytest-asyncio torchvision torchaudio scikit-learn scipy matplotlib

# Frontend dependencies
cd studio
npm install
cd ..
```

---

## 3. Data Ingestion & Provenance Fixtures

The benchmark utilizes procedural industrial anomaly fixtures conforming strictly to official MVTec AD and VisA directory schemas (`train/good`, `val_calibration/good`, `test/good`, `test/defect_name`, `ground_truth/defect_name`).

To generate or verify benchmark fixtures:
```bash
python scripts/prepare_benchmark_data.py
```
This populates:
- `data/processed/mvtec/{bottle, cable, capsule, leather, tile}`
- `data/processed/visa/candle`

---

## 4. End-to-End Execution Sequence

Execute the complete research, training, calibration, and evaluation suite in order:

### Step 1: Model Training & Conformal Calibration (Phases 8 & 11)
Trains VISTA-VLM v4.0 with multi-scale DINOv2 feature aggregation, global-local FiLM fusion, and Squeeze-and-Excitation progressive decoding under multi-task Sobel boundary loss. Calibrates non-conformity threshold $\tau_c$ ($\alpha = 0.05$) and regularized shrinkage covariance matrix $\mathbf{\Sigma}^{-1}$.
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 3 --batch_size 4
```
**Artifacts Generated**:
- `checkpoints/vista_mvtec_bottle.pth` (Model weights, v4.0.0)
- `checkpoints/calibration_mvtec.json` (Conformal threshold $\tau_c$)
- `checkpoints/ood_mvtec.json` (Shrinkage prototype & covariance $\mathbf{\Sigma}^{-1}$)
- `benchmarks/results/vista_results_bottle.json` (Diagnostic test metrics)
- `benchmarks/results/roc_pr_bottle.png` (ROC & Precision-Recall curves)
- `benchmarks/results/calibration_bottle.png` (Reliability diagram)

---

### Step 2: Multi-Track Industrial Benchmark Suite (Phase 16)
Evaluates all 4 strictly segregated evaluation tracks across 6 industrial categories:
- Track 1: In-Domain Normal-Only (PatchCore, PaDiM, VISTA-VLM)
- Track 2: Source-Only Cross-Category Transfer (Bottle $\to$ Unseen categories)
- Track 3: Zero-Shot Multimodal VLM (WinCLIP, AnomalyCLIP)
- Track 4: Few-Normal-Shot Sample Efficiency ($k \in \{1, 2, 4, 8\}$)
```bash
python scripts/run_multi_track_benchmark.py
```
**Artifacts Generated**:
- `benchmarks/results/multi_track_benchmark.json`
- `benchmarks/results/multi_category_benchmark.json` (Studio dashboard feed)

---

### Step 3: Systematic 10-Configuration Ablation Study (Phase 18)
Executes all 10 architectural, loss, and inference policy ablations under standardized conditions:
```bash
python vista_vlm/experiments/run_ablation.py
```
**Artifacts Generated**:
- `benchmarks/results/ablation_study.json`
- `reports/ABLATION_STUDY.md`

---

### Step 4: Industrial Failure Taxonomy & Error Audit (Phase 19)
Audits individual sample inspections, classifying failures into False Alarms, False Passes, Small-Defect Misses, Boundary Leakage, and Localization Shifts:
```bash
python scripts/run_error_analysis.py
```
**Artifacts Generated**:
- `benchmarks/results/error_analysis.json`
- `reports/ERROR_ANALYSIS.md`

---

### Step 5: Complete Unit & Integration Test Suite
Executes all 49 regression and mathematical correctness tests:
```bash
python -m pytest tests -v
```
**Expected Outcome**: 49 passed in ~30s (100% green).

---

### Step 6: Frontend Studio Verification & Launch
Build and serve the production React 19 inspection interface:
```bash
# Build production bundle
npm --prefix studio run build

# Start FastAPI backend (Port 8000)
python -m uvicorn vista_vlm.api.app:app --host 0.0.0.0 --port 8000

# Start Vite dev server (Port 5173)
cd studio && npm run dev
```

---

## 5. Random Seeds & Determinism

For strict mathematical reproducibility:
- PyTorch CPU seed: `42` (`torch.manual_seed(42)`)
- NumPy RNG seed: `42` (`np.random.seed(42)`)
- Synthetic Anomaly Generator seed: `42` (`SyntheticAnomalyGenerator(seed=42)`)
- Few-shot sampling seed: `42`

