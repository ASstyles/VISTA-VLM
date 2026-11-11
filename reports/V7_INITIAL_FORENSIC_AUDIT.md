# VISTA-VLM V7.0 — Initial Forensic Audit & Baseline Qualification Report

**Audit Date:** 2026-11-11  
**Lead Auditor:** Principal Systems Architect & Chief AI Scientist  
**Target Development Branch:** `v7-research-accuracy-reliability`  
**Base Commit:** `d43800488ac17e9ee099c47945a5c893d577f3d8` (v6-research-reliability)  
**Pull Request Under Review:** PR #2  

---

## 1. Executive Summary & Purpose

This forensic audit establishes an unvarnished, mathematically rigorous baseline of the VISTA-VLM codebase prior to the V7.0 architectural and research overhaul. 

VISTA-VLM was originally conceived as a multimodal industrial visual anomaly detection platform combining a visual foundation model (DINOv2), multi-scale feature fusion, progressive decoding, nominal memory bank guidance, and conformal prediction for false-alarm rate control. 

While the V6.0 iteration introduced crucial architectural components (e.g., exact and approximate greedy $k$-center coreset sampling, split conformal prediction scaffolding, and a 9-state operational escalation engine), our forensic investigation reveals **critical scientific flaws in experimental protocol, training-inference divergence, data leakage during calibration, and evaluation gate suppression**.

Most egregiously:
1. **The V6 qualification benchmark reported a 66.67% mean false alarm rate** and an **IoU of 0.3531**, yet **Gate C was marked as `PASS`** due to conditional logic that completely ignored operational false alarm rates and localization quality.
2. **Conformal calibration suffered severe data leakage**: the normal reference memory bank was constructed from `train_loader`, and conformal non-conformity scores were calibrated on that exact same `train_loader`. This caused catastrophic test-time distribution collapse.
3. **Training and inference used disconnected scoring branches**: during training, loss was computed on a neural head, but during inference with a fitted memory bank, the neural head was completely ignored in favor of nearest-neighbor distance and spatial top-$k$ pooling.
4. **The qualification runner evaluated an untrained neural network**: `scripts/run_v6_qualification.py` initialized fresh, randomly initialized decoder and fusion weights and never trained them, evaluating random projection weights against fully fitted baselines (PatchCore and PaDiM).
5. **Dataset provenance was entirely synthetic**: all evaluations were performed on procedural synthetic fixtures in `data/synthetic_fixtures/`, as real datasets (`data/real/`) consisted solely of empty `.gitkeep` markers.

---

## 2. Mandatory V6 Baseline Reproduction

The baseline metrics established in V6.0 (`reports/v6_qualification/v6_qualification_report.json`) are preserved verbatim below as the scientific reference anchor:

| Metric | V6 Reported Result | Evaluation Condition | Scientific Status |
|---|---:|---|---|
| **Macro Image AUROC** | **0.9250 ± 0.0863** | Synthetic Fixtures (6 categories) | Deficient vs Baselines (PatchCore: 0.9847) |
| **Macro Pixel AUROC** | **0.9819 ± 0.0184** | Synthetic Fixtures (6 categories) | Moderate |
| **Macro AUPRO** | **0.8892 ± 0.1441** | Synthetic Fixtures (6 categories) | Deficient vs Baselines (PaDiM: 0.9098) |
| **Macro IoU** | **0.3531 ± 0.2547** | Synthetic Fixtures (6 categories) | **Severe Localization Failure** |
| **Macro Dice** | **0.4793 ± 0.2745** | Synthetic Fixtures (6 categories) | **Severe Localization Failure** |
| **Mean False Alarm Rate (FAR)** | **66.67% ± 51.64%** | Target $\alpha = 0.05$ (5.0%) | **Catastrophic Calibration Collapse** |
| **Gate A (Software Engineering)** | `PASS` | 95 tests passing, lint clean | Valid |
| **Gate B (Research Validity)** | `BLOCKED` | Missing external real datasets | Valid (Correctly identified) |
| **Gate C (Detection Quality)** | `PASS` | Macro metrics checked | **INVALID (Evaluation Logic Bug)** |
| **Gate D (Industrial Pilot)** | `BLOCKED` | No physical hardware deployment | Valid |
| **Gate E (Functional Safety)** | `RESTRICTED_ADVISORY` | Advisory boundary | Valid |

### Track 1 Macro Competitor Comparison (V6 Baseline)

| Model | Macro Image AUROC | Macro Pixel AUROC | Macro AUPRO | Macro IoU | Macro Dice | Avg False Alarm |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore** | **0.9847 ± 0.0374** | 0.9885 ± 0.0195 | 0.8969 ± 0.1577 | **0.6448** | **0.7741** | **0.00%** |
| **PaDiM** | 0.9708 ± 0.0714 | **0.9919 ± 0.0118** | **0.9098 ± 0.1368** | 0.6370 | 0.7689 | 10.00% |
| **VISTA-VLM V6.0** | 0.9250 ± 0.0863 | 0.9819 ± 0.0184 | 0.8892 ± 0.1441 | 0.3531 | 0.4793 | 66.67% |

---

## 3. Comprehensive Codebase Inspection & Forensic Findings

### 3.1 Model Architecture & Inference Flow (`vista_vlm/models/`)

#### A. Training vs Inference Scoring Mismatch (`vista_model.py:378-396`)
```python
if self.scoring_mode == "memory_max" and raw_mem_scores is not None:
    image_score = cal_mem_score
elif self.scoring_mode == "memory_topk":
    image_score = topk_spatial_score
elif ...:
    ...
else:  # "hybrid"
    if raw_mem_scores is not None:
        # 70% calibrated memory distance + 30% spatial top-k map score
        image_score = 0.70 * cal_mem_score + 0.30 * topk_spatial_score
    else:
        image_score = self.global_head_weight * head_score + (1.0 - self.global_head_weight) * topk_spatial_score
```
- **Finding**: During model training (`scripts/train_and_calibrate.py`), the memory bank is not yet fitted (`raw_mem_scores is None`). Therefore, gradient updates train `self.global_head` and `self.decoder`. 
- **Flaw**: Once `fit_memory_bank()` is called, `raw_mem_scores` becomes active. The model shifts branches and completely ignores `self.global_head`! The parameters optimized during training are discarded during inference.
- **Qualification Runner Abnormality**: In `scripts/run_v6_qualification.py:144-158`, the qualification script instantiates `VISTAAnomalyModel` and directly fits the memory bank without running *any* parameter training for the fusion and decoder networks. VISTA was evaluated with purely random projection weights.

#### B. Memory Bank $k$-NN Truncation (`memory_bank.py:301-303`)
```python
min_dists, _ = torch.topk(dists, k=self.k_neighbors, dim=1, largest=False)
dist_map = min_dists[:, 0].reshape(1, 1, gh, gw)
dist_maps.append(dist_map)
```
- **Finding**: While `k_neighbors` is accepted as a parameter (defaulting to 1 or 3), `compute_distance_map()` explicitly indexes `min_dists[:, 0]`, discarding all remaining $k-1$ neighbors. No multi-neighbor aggregation (mean, distance-weighted, or harmonic) was actually executed.

#### C. Unchecked Memory Bank Checksum Failure (`memory_bank.py:348-350`)
```python
if self.sha256 and curr_hash != self.sha256:
    logger.warning(f"Memory bank SHA-256 digest mismatch! Expected {self.sha256}, got {curr_hash}")
```
- **Finding**: When loading a corrupted or tampered memory bank, the method issues a non-blocking logger warning and continues execution. In a production industrial system, this must raise an exception and fail closed (`ModelState.INVALID`).

#### D. Non-Robust Reference Distribution Scaling (`memory_bank.py:233-255`)
- Reference statistics were computed by sampling the first $N \le 2000$ training patches and using `min` and `max`. Heuristic `scale_range = (ref_max - ref_min) * 1.5` is brittle to extreme outlier patches in normal training samples.

---

### 3.2 Calibration & Selective Prediction (`vista_vlm/calibration/`)

#### A. Data Leakage in Conformal Calibration (`run_v6_qualification.py:158-169`)
```python
# Fit memory bank
vista_model.fit_memory_bank(train_loader, device=device, ...)

# Conformal calibration on train normal split (LEAKAGE!)
conformal = ConformalCalibrator(target_false_alarm_rate=cfg["calibration"]["target_false_alarm_rate"])
train_scores = []
with torch.no_grad():
    for b in train_loader:
        out = vista_model(b["image"].to(device))
        train_scores.extend(out["image_score"].cpu().numpy().tolist())
conformal_thresh = conformal.fit(train_scores, category=cname)
```
- **Mathematical Root Cause of 66.67% FAR**: The memory bank was populated from `train_loader`. Evaluating `train_loader` against its own coreset yields near-zero nearest-neighbor distances. 
- Calibrating the $(1-\alpha)$ quantile on training scores produced an artificially tiny threshold $\tau_c$.
- At test time, unseen normal test images had natural sampling variance and higher distances, causing two-thirds of normal items to be falsely flagged as defective.

#### B. Inconsistent Decision Rule at Threshold (`conformal.py:164`)
```python
is_anomaly = bool(score >= threshold or p_val <= self.alpha)
```
- For upper-tail conformal prediction with $n$ samples, rank $k = \lceil(n+1)(1-\alpha)\rceil$, and threshold $\tau_c = s_{(k)}$.
- For an observation exactly at threshold $s = \tau_c$, the count of calibration scores $\ge s$ is at least $n - k + 1$. 
- The upper-tail p-value is $p(s) = \frac{1 + \sum_{i=1}^n \mathbf{1}(s_i \ge s)}{n+1} \ge \frac{n - k + 2}{n + 1} > \alpha$.
- Combining `score >= threshold` with `p_val <= alpha` with an `or` operator yields conflicting mathematical decisions.

#### C. Inadequate Checksum Verification in Calibrator (`conformal.py:184-194`)
- `ConformalCalibrator.load()` deserializes JSON and stores `checksums`, but never verifies that the stored SHA-256 digest matches the recomputed digest of `calibration_scores`. Corrupted files load without error.

#### D. Selective Prediction Tuning on Synthetic Artifacts (`selective.py:100-152`)
- While Wilson intervals and exact Clopper-Pearson binomial bounds were correctly implemented, selective prediction operating-point tuning used synthetic perturbations rather than an independent labelled validation partition.
- Defect escape guarantees cannot be asserted on synthetic fixtures.

---

### 3.3 Dataset Loaders & Provenance (`vista_vlm/data/`)

#### A. VisA Random Split Non-Compliance (`visa.py:71-80`)
```python
normal_files = sorted(glob.glob(os.path.join(found_normal[0], "*.*")))
rng.shuffle(normal_files)
# Split normal images into train (70%), val_calibration (15%), test normal (15%)
```
- **Finding**: `VisADataset` randomly shuffled normal images rather than parsing official VisA split metadata (e.g. `image_list.csv` or standard train/test splits).
- This violates standard benchmark comparability and makes reproduction impossible across external publications.

#### B. Manifest Manager Split Coverage (`manifest_manager.py:82-245`)
- The integrity checker supported only binary `train` vs `test` splits. It lacked explicit tracking for the 4-partition protocol:
  1. `train` (normal only)
  2. `val_calibration` (normal only holdout)
  3. `val_tuning` (labelled normal and defectives)
  4. `test` (final blind holdout)

#### C. Absolute Lack of Real Dataset Assets
- `data/real/` contains only placeholder `.gitkeep` files. No external MVTec AD, VisA, BTAD, or MPDD images are present on local disk.
- All baseline numbers are derived strictly from procedural synthetic fixtures.

---

### 3.4 Evaluation & Qualification Runner (`vista_vlm/evaluation/`)

#### A. The Gate C Suppression Bug (`run_v6_qualification.py:286-290`)
```python
gate_c_pass = bool(
    vista_macro["image_auroc"]["mean"] >= 0.90 and
    vista_macro["pixel_auroc"]["mean"] >= 0.95 and
    vista_macro["aupro"]["mean"] >= 0.80
)
```
- **Finding**: Gate C declared `PASS` while completely omitting:
  - False alarm rate check (which was 66.67%, far exceeding the 5% target).
  - IoU check (which was 0.3531).
  - Dice check (which was 0.4793).
- This masked severe localization and operational reliability failures.

#### B. Threshold Bleed between Image and Pixel Levels (`run_v6_qualification.py:176`)
```python
vista_res = evaluate_vista_model(vista_model, test_loader, device=device, calibrated_threshold=conformal_thresh)
```
- In `benchmark_runner.py:51`:
```python
pix_metrics = compute_pixel_metrics(y_masks, y_pred_maps, threshold=calibrated_threshold)
```
- The image-level conformal threshold ($\approx 0.90$) was passed directly as the binarization cutoff for pixel anomaly maps!
- In contrast, PatchCore and PaDiM were evaluated with `threshold=None` (which dynamically selects the optimal F1 threshold).
- This artificially crushed VISTA-VLM's pixel IoU and Dice scores.

#### C. Statistical Representation Fallacy (`metrics.py:230-245`)
- The reported confidence intervals in V6 (e.g., `[0.8344, 1.0000]` on Image AUROC) represented the sample standard deviation across 6 categories divided by $\sqrt{6}$.
- This was mischaracterized in documentation as an image-level statistical confidence interval.

---

### 3.5 API, Application Security & Infrastructure (`vista_vlm/api/`)

#### A. Overly Permissive CORS Configuration (`app.py:25-31`)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- **Vulnerability**: Combining `allow_origins=["*"]` with `allow_credentials=True` violates RFC standards and creates high-severity cross-origin data exposure risks in industrial intranets.

#### B. Exception Information Disclosure (`app.py:33-39`)
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": request.url.path}
    )
```
- Internal stack traces and file paths are directly leaked to API consumers.

#### C. Hardcoded Readiness & Inspection Assertions (`pipeline.py:205-206`)
```python
escalation_eval = self.selective_predictor.evaluate_decision(
    ...,
    image_quality_ok=True,
    model_ready=True,
    ...
)
```
- In `routes.py:/health/ready`, the check does not verify whether model checkpoints, fitted memory banks, or calibration artifacts are actually valid on disk.

---

## 4. Discrepancies Between Documentation and Code

| Documented Claim (README / Docs) | Actual Code Implementation | Forensic Finding |
|---|---|---|
| *"Rigorous Split-Conformal False Alarm Rate Guarantees (<= 5%)"* | `run_v6_qualification.py:165` trains and calibrates on identical data; empirical FAR is **66.67%**. | **Violated** (Data leakage during calibration) |
| *"Certified Pixel Localization with IoU >= 0.60, Dice >= 0.75"* | Reported Macro IoU is **0.3531**, Macro Dice is **0.4793**. | **Violated** (Severe localization shortfall) |
| *"Gate C: Passed Detection Quality"* | Evaluated using code that omits false-alarm and localization criteria. | **Violated** (Suppressed failure gate) |
| *"Calibrated Memory Residual Fusion with Trained Global Head"* | `vista_model.py:390` completely bypasses the trained global head whenever memory bank is fitted. | **Violated** (Architectural disconnect) |
| *"Official VisA Split Compliance"* | `visa.py:71` uses `rng.shuffle()` random subsampling. | **Violated** (Non-standard splits) |
| *"Cryptographic Artifact Verification"* | `conformal.py` and `memory_bank.py` either ignore checksum mismatches or issue warnings. | **Violated** (Fail-open loading) |

---

## 5. Remediation Plan & Phase Alignment for V7.0

1. **Phase 2 (Scientific Experiment Design)**:
   - Establish strict 4-way independent partitioning: Train (normal), Calibration (normal), Tuning (normal + defect), Test (blind).
   - Implement official VisA split parser with CSV validation and fallback classification.
   - Upgrade `DatasetIntegrityChecker` to eliminate leakage across all four splits.
2. **Phase 3 (Training & Inference Unification)**:
   - Harmonize scoring branch so training loss and deployed inference share identical formulations.
   - Train fusion and decoder components with genuine optimization before memory bank finalization.
3. **Phase 4 (Anomaly Scoring & Memory Guidance)**:
   - Repair $k$-NN distance map aggregation across all configured neighbors.
   - Implement robust statistical reference estimators (median, MAD, high normal quantiles).
   - Optimize score fusion weights on holdout validation data.
4. **Phase 5 (Mathematically Correct Calibration)**:
   - Enforce exact split-conformal rank $k = \lceil(n+1)(1-\alpha)\rceil$ with sample size rejection.
   - Unify threshold and p-value decision rule.
   - Implement atomic saves and strict checksum validation.
5. **Phase 6 (Selective Prediction & Risk Bounds)**:
   - Evaluate selective prediction only on independent labelled tuning sets.
   - Report exact Clopper-Pearson / Wilson binomial upper bounds.
6. **Phase 7 (Benchmark Suite & Gate Hardening)**:
   - Re-evaluate all tracks.
   - Repair Gate C logic to strictly enforce $\le 5\%$ FAR and minimum localization thresholds.
   - Implement per-image bootstrap confidence intervals.
7. **Phases 8-12 (Production Lifecycle, Security, Frontend & CI)**:
   - Enforce fail-closed state machine (`UNINITIALIZED`, `LOADING`, `READY`, `INVALID`).
   - Secure API (restrict CORS, suppress stack traces, real readiness checks).
   - Verify TypeScript frontend build and CI Docker workflow.

---
*Certified by the VISTA-VLM V7.0 Research & Principal Engineering Council.*

