# V7.1 Benchmark Evaluation Protocol & Fair Baseline Comparison Audit

**Author**: VISTA-VLM Core Research Team  
**Branch**: `v7-research-validation-hardening`  
**Date**: November 2026  
**Artifact**: `reports/V7_1_BENCHMARK_PROTOCOL_AUDIT.md`  
**Modules Audited**: `vista_vlm/evaluation/benchmark_runner.py`, `vista_vlm/evaluation/metrics.py`, `vista_vlm/data/mvtec.py`, `vista_vlm/data/visa.py`  

---

## 1. Executive Summary & Objective

In visual anomaly detection research, benchmarking flaws are widespread:
1. **Test-Set Optimization of Baseline Thresholds**: In V7.0, `evaluate_baseline_model()` called `compute_image_metrics()` and `compute_pixel_metrics()` with `threshold=None`, which performed an argmax F1 search directly across the untouched test labels and masks. In contrast, VISTA used split-conformal thresholds tuned on holdout data.
2. **Partition Leakage & Protocol Ambiguity**: Custom carve-outs of standard datasets (e.g., taking 20% of MVTec test samples for tuning) were previously described ambiguously, risking confusion with official untouched benchmark splits.
3. **Coupling of Threshold-Free and Threshold-Dependent Metrics**: Failure to strictly separate rank-based performance (AUROC, Average Precision, AUPRO) from operating-point dependent metrics (F1, Precision, Recall, IoU, Dice).

This audit establishes the hardened V7.1 benchmark evaluation protocol ensuring scientific defensibility and mathematical fairness.

---

## 2. Disjoint Four-Partition Protocol Architecture

Every evaluation category is strictly partitioned into four disjoint subsets with deterministic sample identity:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Industrial Dataset Pool                         │
└────────────────────────────────────────────────────────────────────────┘
                   │                                  │
         Nominal Normal Images                Defective & Mixed Images
                   │                                  │
         ┌─────────┴─────────┐              ┌─────────┴─────────┐
         ▼                   ▼              ▼                   ▼
    ┌─────────┐        ┌───────────┐   ┌─────────┐        ┌───────────┐
    │  Train  │        │Val-Calib  │   │Val-Tune │        │   Test    │
    │(Nominal)│        │ (Normal)  │   │(Labelled│        │ (Untouched│
    │         │        │           │   │ Norm+Def│        │ Blind Eval│
    └─────────┘        └───────────┘   └─────────┘        └───────────┘
         │                   │              │                   │
  Model Training &    Split-Conformal  Operating Point,   Final Untouched
  Memory Coreset       Threshold tau   Selective Band &   Evaluation Only
   Construction                         Pixel Thresh
```

### Partition Responsibilities & Access Restrictions

| Partition | Composition | Role in Lifecycle | Access Constraints |
|:---|:---|:---|:---|
| **`train`** | Nominal Normal Only | Backbone feature extraction, nominal memory coreset fitting, decoder training | Never sees anomalous samples |
| **`val_calibration`** | Nominal Normal Only | Split-conformal calibration ($k = \lceil(n+1)(1-\alpha)\rceil$), OOD feature profiling | Independent normal samples; never used for backprop |
| **`val_tuning`** | Labelled (Normal + Defective) | Threshold calibration ($\tau_{\text{img}}, \tau_{\text{pix}}$), selective uncertainty band optimization ($\delta^*$) | Never used for training model weights; independent of test |
| **`test`** | Labelled (Normal + Defective) | Final scientific metric reporting | Strictly untouched; zero influence on any parameter or threshold |

---

## 3. Fair Baseline Comparison Protocol

### 3.1 Audited Baseline Models
- **PatchCore** (Roth et al.): Patch-level memory bank with greedy k-Center coreset downsampling (ratio 0.10) and k-NN distance scoring ($k=3$).
- **PaDiM** (Defard et al.): Multi-scale multivariate Gaussian patch modeling with dimensional reduction ($d=100$).
- **WinCLIP** (Jeong et al.): Zero-shot multi-scale text-visual associative classification and localization.
- **AnomalyCLIP** (Zhou et al.): Object-agnostic learned prompt tuning for zero-shot defect identification.

### 3.2 Elimination of Baseline Test-Label Snooping
To eliminate the asymmetric advantage where baselines previously optimized operating thresholds on test labels:
1. **Holdout Threshold Calibration**:
   `calibrate_baseline_thresholds()` evaluates baseline models exclusively on `val_tuning` (or `val_calibration`), finding:
   $$\tau^*_{\text{img}} = \arg\max_{\tau} F_1(\text{scores}_{\text{tune}}, y_{\text{tune}}; \tau)$$
   $$\tau^*_{\text{pix}} = \arg\max_{\tau} F_1(\text{maps}_{\text{tune}}, M_{\text{tune}}; \tau)$$
2. **Blind Evaluation**:
   `evaluate_baseline_model()` accepts `calibrated_threshold=\tau^*_{\text{img}}` and `pixel_threshold=\tau^*_{\text{pix}}`.
   All threshold-dependent metrics (F1, Precision, Recall, IoU, Dice) are computed strictly at these holdout-tuned operating points.
3. **Threshold-Free Isolation**:
   AUROC, Average Precision, and AUPRO are evaluated across all continuous prediction ranks and documented independently from threshold-dependent numbers.

---

## 4. Benchmark Dataset Protocol Integrity

### 4.1 MVTec AD Protocol Clarification
- **Official Benchmark Protocol**: MVTec AD official test sets contain untouched defects and normals. When evaluating official benchmark scores without a separate tuning source, threshold-free metrics (AUROC, AP, AUPRO) are reported.
- **Custom Research Protocol (4-Partition)**: When holdout tuning is required from the test pool, MVTec test samples are deterministically partitioned into `val_tuning` (20%) and `test` (80%). The sample metadata explicitly records `split_protocol="custom_research_4partition"` to eliminate ambiguity.
- **Synthetic Development Protocol**: Procedural benchmark generators record `split_protocol="procedural_synthetic_benchmark_fixture"`.

### 4.2 VisA Protocol Compliance
- Official split definitions are parsed from `image_list.csv` or `split_csv/*.csv`.
- CSV columns (`image_path`, `mask_path`, `label`, `split`) are strictly validated against on-disk file checksums.
- Image-mask dimension correspondence is enforced (height and width verified via PIL headers before loading).
- If official CSV files are missing, the loader records `split_protocol="deterministic_fallback"` in the dataset manifest, preventing silent protocol substitution.

