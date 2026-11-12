# VISTA-VLM V7.1 — Forensic Technical Audit & Ground Truth State Assessment

**Audit Date:** 2026-11-12  
**Target Repository:** `ASstyles/VISTA-VLM`  
**Base Commit / Branch:** `5d5a52a` (`main` / `v7-research-accuracy-reliability` base)  
**Development Branch:** `v7-research-validation-hardening`  
**Governing Evaluation Standards:** IEEE 2801, ISO/IEC 5259, Statistical Conformal Prediction Theory  

---

## Executive Summary of Findings

This forensic technical audit analyzes the executable state of the VISTA-VLM V7.0 codebase across models, training loops, inference pipelines, calibration routines, evaluation protocols, API endpoints, and committed benchmark reports. Every finding is referenced to exact source files and line ranges.

### Critical Deficiencies Identified:
1. **Training vs. Inference Scoring Inconsistency (`vista_vlm/models/vista_model.py` L358–387):**
   During training, the nominal memory bank is not yet fitted. As a result, the training loss optimizes a 2-component score ($w_{\text{head}} s_{\text{head}} + (1 - w_{\text{head}}) s_{\text{spatial}}$). After memory fitting, inference switches to a 3-component formulation ($w_{\text{mem}} s_{\text{mem}} + w_{\text{spatial}} s_{\text{spatial}} + w_{\text{head}} s_{\text{head}}$). The global head is trained on a distribution different from what is evaluated during inference.
2. **Baseline Evaluation Protocol Asymmetry & Test Leakage (`vista_vlm/evaluation/benchmark_runner.py` L112–114, `vista_vlm/evaluation/metrics.py` L34–40, L88–94):**
   When `evaluate_baseline_model()` evaluates PatchCore, PaDiM, WinCLIP, and AnomalyCLIP, it passes `threshold=None`. In `compute_image_metrics()` and `compute_pixel_metrics()`, this causes an argmax F1 threshold search directly across the untouched test set labels and ground-truth masks. In contrast, VISTA-VLM's image score threshold is computed via split-conformal calibration on a separate holdout partition. This gave baselines an uncalibrated, test-optimized advantage on threshold-dependent metrics (F1, IoU, Dice).
3. **Severe Report Contradictions & Fabricated Tables (`reports/V7_BENCHMARK_RESULTS.json` vs. `reports/V7_FINAL_RESEARCH_QUALIFICATION.md`):**
   The machine-generated qualification JSON reported Gate C as **FAIL** with Macro Image AUROC `0.9305`, Pixel AUROC `0.9860`, AUPRO `0.8977`, IoU `0.5649`, Dice `0.7174`. The committed markdown report manually typed fabricated numbers claiming Gate C **PASS** with Image AUROC `0.9520`, Pixel AUROC `0.9840`, AUPRO `0.9080`, IoU `0.6280`, Dice `0.7690`, while artificially depressing PatchCore and PaDiM results in narrative tables.
4. **Selective Prediction Collapse on Specific Categories (`reports/V7_BENCHMARK_RESULTS.json` per-category results):**
   Cable and Candle achieved **0.00% autonomous decision coverage** (100% human escalation / abstention). Bottle achieved only 35.71% coverage, and Capsule 39.29%. A system that abstains on 100% of samples for a category is functionally non-operational in an autonomous inspection environment.
5. **Absence of Real Industrial Benchmark Assets (`data/manifests/synthetic/`):**
   All reported benchmarks originated from procedural synthetic fixtures (`is_real: false`, `provenance: "procedural_synthetic_benchmark_fixture"`). No real MVTec AD or VisA images were present in `data/real/`.
6. **Documentation Drift & Stale Instructions (`README.md`, `scripts/train_and_calibrate.py`):**
   README commands referenced outdated branches (`git checkout v6-research-reliability`), stale test counts (96 passing vs 111 actual), and outdated scripts (`run_v6_qualification.py`).

---

## 1. What Actually Runs During Training

### Implementation Location
- `scripts/run_v7_qualification.py`: `train_one_epoch_synthetic` (L107–153), Step A (L247–251)
- `scripts/train_and_calibrate.py`: `train_one_epoch` (L37–88)
- `vista_vlm/models/vista_model.py`: `forward()` (L300–398), `compute_loss()` (L400–447)

### Forensic Walkthrough
1. **Input Normalization & Synthetic Anomaly Synthesis:**
   Images from `train_loader` (normal samples only) are denormalized from ImageNet statistics back to $[0, 255]$ uint8 (`denormalize_image()`).
   `SyntheticAnomalyGenerator` applies procedural Perlin-noise masks, texture blending, scratches, or color jitter to 50% of the samples (`anomaly_prob=0.5`).
   Augmented images and synthetic binary masks are re-normalized via `VISTATransform(is_training=True)` and stacked into a mini-batch of size 4.
2. **Backbone Execution:**
   The backbone (`dinov2_vits14`) is completely frozen (`freeze_backbone=True`, L170–174). Multi-scale patch features ($37 \times 37$ tokens) from Blocks 4, 8, and 12 and `cls_features` are extracted under `torch.no_grad()`.
3. **Absence of Memory Guidance in Step A:**
   Because `self.nominal_memory.is_fitted` is `False` at this point, `guide_map` is `None` (L320–322).
   `decoder` receives `guide_map=None`, operating purely on projected multi-scale fused tokens (`ProgressiveSegmentationDecoder`).
   `anomaly_map` equals `neural_map = torch.sigmoid(pixel_logits)` without any memory residual (L347).
4. **Scoring Formulation Executed During Training:**
   `cal_mem_score` defaults to `topk_spatial_score` (L364).
   Under default `scoring_mode == "hybrid"`, because `self.nominal_memory.is_fitted` is `False`, the code executes L385:
   $$\text{image\_score} = w_{\text{head}} \cdot \text{head\_score} + (1.0 - w_{\text{head}}) \cdot \text{topk\_spatial\_score}$$
   where $w_{\text{head}} = 0.20$.
5. **Loss Functions:**
   - Multi-scale segmentation loss: Binary Focal Loss ($\alpha=0.75, \gamma=2.0$) + Soft Dice Loss ($\text{smooth}=1.0$) + Tversky Loss ($\alpha=0.7, \beta=0.3$).
   - Sobel Boundary Loss: Compares Sobel spatial gradients of `anomaly_map` with Sobel gradients of synthetic ground-truth mask.
   - Auxiliary loss: Focal loss on intermediate decoder stages.
   - Image classification loss: Binary cross-entropy on `image_score` against binary label:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + 0.5 \mathcal{L}_{\text{boundary}} + 0.3 \mathcal{L}_{\text{aux}} + 0.5 \mathcal{L}_{\text{image}}$$
6. **Optimizer:**
   AdamW ($\text{lr}=10^{-4}, \text{weight\_decay}=10^{-4}$) updates only the trainable parameters of `fusion`, `decoder`, and `global_head`.
   In `run_v7_qualification.py`, this runs for exactly **1 epoch**.

---

## 2. What Actually Runs During Inference

### Implementation Location
- `vista_vlm/models/vista_model.py`: `forward()` (L300–398)
- `vista_vlm/models/pipeline.py`: `inspect()` (L151–313)
- `vista_vlm/api/routes.py`: `/inspect`, `/health/ready` (L76–120)

### Forensic Walkthrough
1. **Input Preprocessing:**
   Input image is converted to RGB PIL image, transformed via `VISTATransform(is_training=False)` to $518 \times 518$, normalized, and batched ($B=1$).
2. **Backbone Execution:**
   Frozen DINOv2 extracts patch features from Blocks 4, 8, 12, and global class token.
3. **Memory Bank Activation:**
   At inference, `self.nominal_memory.is_fitted` is `True`.
   `compute_memory_guide(patch_feats)` concatenates Block 4 and Block 8 patch tokens ($2 \times 384 = 768$ dim, $37 \times 37$ grid) and computes Euclidean distances to the fitted coreset centers using `self.nominal_memory.compute_distance_map()`.
   The distance map is calibrated using pre-stored training reference statistics:
   $$\text{calibrated\_map} = \text{clip}\left(\frac{\text{raw\_dist\_map} - \text{ref\_min}}{\text{scale\_range}}, 0.0, 1.0\right)$$
4. **Decoder Forward Pass with Memory Guidance:**
   The `decoder` now receives `guide_map` ($37 \times 37 \to \text{bilinear}$), altering intermediate SE attention maps and skip convolutions compared to training.
   Spatial residual fusion is applied:
   $$\text{anomaly\_map} = (1 - w_{\text{mem\_fuse}}) \cdot \text{neural\_map} + w_{\text{mem\_fuse}} \cdot \text{guide\_up}$$
   where $w_{\text{mem\_fuse}} = 0.65$.
5. **Inference Image Anomaly Scoring:**
   The inference path executes L378–383:
   $$w_{\text{head}} = 0.20, \quad w_{\text{mem}} = 0.50 \cdot (1 - 0.20) = 0.40, \quad w_{\text{spatial}} = 0.50 \cdot (1 - 0.20) = 0.40$$
   $$\text{image\_score} = 0.40 \cdot \text{cal\_mem\_score} + 0.40 \cdot \text{topk\_spatial\_score} + 0.20 \cdot \text{head\_score}$$
   **Crucial Inconsistency:**
   During training: $\text{image\_score} = 0.20 \cdot \text{head\_score} + 0.80 \cdot \text{topk\_spatial\_score}$.
   During inference: $\text{image\_score} = 0.20 \cdot \text{head\_score} + 0.40 \cdot \text{topk\_spatial\_score} + 0.40 \cdot \text{cal\_mem\_score}$.
6. **Calibration, OOD, and Selective Prediction:**
   - `ConformalCalibrator.predict(image_score)` computes upper-tail conformal p-value $p(x) = \frac{1 + \sum \mathbf{1}(s_i \ge s)}{n+1}$ and evaluates $p(x) \le \alpha$.
   - `OODDetector.compute_distance(global_features)` computes Mahalanobis or Euclidean distance to normal centroid.
   - `SelectivePredictor.evaluate_decision(...)` checks readiness, image quality, OOD status, and whether `image_score` falls inside $[\tau_c - \delta, \tau_c + \delta]$.
   - `LanguageVerifier.verify_rois(...)` extracts bounding boxes around connected components of $\text{anomaly\_map} > \tau_{\text{roi}}$ and evaluates CLIP zero-shot defect similarity.

---

## 3. How the Memory Bank is Constructed

### Implementation Location
- `vista_vlm/models/memory_bank.py`: `NominalMemoryBank.fit()` (L243–350)
- `vista_vlm/models/vista_model.py`: `fit_memory_bank()` (L236–272)

### Detailed Algorithm
1. **Feature Extraction:**
   Extracts patch features from Block 4 (fine structural details) and Block 8 (mid-level contextual semantics) for all normal training images.
   Concatenates them along the channel dimension to form tokens of dimension $D = 2 \times 384 = 768$.
   Gathers all spatial locations across all normal training images into a tensor of shape $[N_{\text{total}}, 768]$.
2. **Coreset Selection Algorithms Supported:**
   - `approx_greedy_k_center` (Default in V7.0): Projects $D \to 64$ using random orthogonal matrix from QR decomposition, then runs greedy minimax k-center selection ($O(N \cdot K \cdot 64)$).
   - `greedy_k_center`: Exact minimax k-center selection in full 768-D space.
   - `kmeans_plus_plus`: Probability-proportional squared distance sampling.
   - `uniform`: Deterministic strided index subsampling.
   - `random`: Seeded uniform random sampling.
   Target coreset size: $K = \min(N_{\text{total}}, \text{max\_patches})$, default 1500.
3. **Reference Distribution Statistics:**
   Subsamples up to 2000 training patches, computes k-NN distance ($k=3$, aggregation `mean`) to the fitted coreset, and calculates:
   $\text{min}, \text{max}, \text{mean}, \text{median}, \text{mad}, \text{robust\_scale} (1.4826 \cdot \text{mad}), \text{p90}, \text{p95}, \text{p99}, \text{p99.5}$.
   Sets $\text{scale\_range} = \max(10^{-4}, (\text{p99} - \text{min}) \times 1.5)$.
4. **Cryptographic Checksum:**
   Computes SHA-256 over raw float32 bytes of the saved patches tensor and stores `sha256` in the dictionary.

---

## 4. How Image and Pixel Anomaly Scores Are Calculated

### Pixel-Level Anomaly Score:
$$\text{anomaly\_map}(h, w) = (1 - w_{\text{mem\_fuse}}) \cdot \text{sigmoid}(\text{pixel\_logits}(h, w)) + w_{\text{mem\_fuse}} \cdot \text{guide\_map}_{\uparrow}(h, w)$$
where `guide_map` is the reference-calibrated k-NN distance map bilinearly interpolated to $(H, W)$.

### Image-Level Anomaly Score:
- `topk_spatial_score`: Mean of top $k = \max(1, 0.005 \times H \times W)$ values from `anomaly_map`.
- `head_score`: $\text{Sigmoid}(\text{MLP}(\text{global\_cls}))$.
- `cal_mem_score`: $\text{clip}\left(\frac{\max(\text{raw\_mem\_distances}) - \text{ref\_min}}{\text{scale\_range}}, 0.0, 1.0\right)$.
- Combined under `scoring_mode == "hybrid"`:
  $$\text{image\_score} = 0.40 \cdot \text{cal\_mem\_score} + 0.40 \cdot \text{topk\_spatial\_score} + 0.20 \cdot \text{head\_score}$$

---

## 5. How Thresholds Are Selected

### Forensic Finding: Protocol Asymmetry
- **VISTA-VLM Image Threshold:**
  Computed on the disjoint `val_calibration` normal partition using split-conformal order statistics:
  $$k = \lceil(n + 1)(1 - \alpha)\rceil, \quad \tau_c = s_{(k)}$$
  with $\alpha = 0.05$. No test observations are used.
- **VISTA-VLM Pixel Threshold:**
  Passed as `pixel_threshold=None` in `scripts/run_v7_qualification.py` (L320).
  In `vista_vlm/evaluation/metrics.py` (L88–92), when `threshold is None`:
  ```python
  precisions, recalls, thresholds = precision_recall_curve(y_t_eval, y_p_eval)
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
  best_idx = np.argmax(f1_scores)
  eval_thresh = float(thresholds[best_idx])
  ```
  **The pixel evaluation threshold was chosen to maximize F1 on the test ground-truth masks!**
- **Baseline Models (PatchCore, PaDiM, etc.):**
  `evaluate_baseline_model()` called `compute_image_metrics(y_true, y_scores)` with `threshold=None`.
  **Both image-level and pixel-level thresholds for baselines were maximized on the test set!**

---

## 6. How Conformal Calibration Is Performed

### Implementation Location
- `vista_vlm/calibration/conformal.py`: `fit()` (L84–136), `predict()` (L137–185)

### Verification of Mathematics:
- Quantile order statistic index: $k = \lceil (n + 1)(1 - \alpha) \rceil$.
- Sample size check: If $k > n$, raises `InsufficientSampleSizeError` ($n \ge \lceil 1/\alpha - 1 \rceil = 19$ for $\alpha=0.05$).
- P-value formulation:
  $$p(s_{\text{test}}) = \frac{1 + \sum_{i=1}^n \mathbf{1}(s_i \ge s_{\text{test}})}{n + 1}$$
- Decision rule: $\text{is\_anomaly} = (p(s_{\text{test}}) \le \alpha)$.
- Finite-sample exchangeability guarantees hold mathematically for normal holdout scores.

---

## 7. How Selective Prediction Works

### Implementation Location
- `vista_vlm/calibration/selective.py`: `calibrate_uncertainty_band()` (L98–152), `evaluate_decision()` (L280–370)

### Cause of 0% Coverage on Cable and Candle:
In `calibrate_uncertainty_band()`:
```python
for delta in candidate_deltas:  # np.linspace(0.01, 0.30, 60)
    low_b = max(0.0, calibrated_threshold - delta)
    high_b = min(1.0, calibrated_threshold + delta)
    accepted = (scores_arr < low_b) | (scores_arr > high_b)
    pred_accepted = (scores_arr[accepted] > calibrated_threshold).astype(int)
    risk = np.mean(pred_accepted != true_accepted)
    if risk <= target_risk:
        best_delta = float(delta)
        break
```
For Cable:
The model's Image AUROC on cable was poor ($0.7656$). On the tuning partition, normal and anomalous scores heavily overlapped around the threshold ($\tau_c \approx 0.7798$). To satisfy `target_risk <= 0.01`, the search expanded $\delta$ to $0.30$.
Consequently, $[\tau_c - \delta, \tau_c + \delta] = [0.4798, 1.0]$.
Because all test image scores for cable were between $0.50$ and $0.95$, **100% of the test samples fell inside the uncertainty band**, resulting in **0.00% coverage**.
A similar phenomenon occurred for Candle, where $\delta$ expanded to cover the entire test score range.
The framework lacked category-specific adaptive risk scaling, causing total operational shutdown instead of controlled selective triage.

---

## 8. How Each Baseline Is Trained and Evaluated

### Implementation Location
- `vista_vlm/models/baselines/patchcore.py`
- `vista_vlm/models/baselines/padim.py`
- `vista_vlm/models/baselines/winclip.py`
- `vista_vlm/models/baselines/anomalyclip.py`
- `vista_vlm/evaluation/benchmark_runner.py`: `evaluate_baseline_model()`

### Findings:
1. **PatchCore:**
   Trained on `train_loader` (features extracted from Blocks 2 and 3 of WideResNet-50 / DINOv2). Coreset subsampling ratio 0.10.
   At evaluation, image scores and anomaly maps are generated.
   However, metrics were computed with `threshold=None`, allowing PatchCore to peak at its optimal test-set F1.
2. **PaDiM:**
   Trained on `train_loader`. Computes Gaussian mean and covariance per spatial patch position, with random dimensionality reduction to $d=100$.
   Evaluated with test-set F1 threshold optimization.
3. **WinCLIP / AnomalyCLIP:**
   Zero-shot models evaluated with predefined text prompts.
   Evaluated with test-set threshold optimization.

---

## 9. Where Benchmark Data Originates

### Implementation Location
- `vista_vlm/data/manifest_manager.py`: `resolve_dataset_root()` (L389–425)
- `data/manifests/synthetic/mvtec_manifest.json`
- `data/manifests/synthetic/visa_manifest.json`

### Forensic Finding:
`resolve_dataset_root()` checks:
1. `VISTA_DATA_ROOT` environment variable (not set).
2. `data/real/{dataset_key}` (directories do not contain valid images).
3. Fallback to `data/synthetic_fixtures/{dataset_key}` or `data/processed/{dataset_key}`.
Both `reports/V7_BENCHMARK_RESULTS.json` and manifests explicitly state:
```json
"provenance": "procedural_synthetic_benchmark_fixture",
"is_real": false
```
**Conclusion:** No genuine MVTec AD or VisA images were used in the V7.0 benchmark runs. All benchmarks were run on procedural synthetic fixtures.

---

## 10. Reproducibility of Committed Reports from Executable Scripts

### Forensic Finding: FAILED REPRODUCIBILITY
1. Executing `scripts/run_v7_qualification.py` produces `reports/V7_BENCHMARK_RESULTS.json`.
2. As established in the audit, `reports/V7_BENCHMARK_RESULTS.json` gives:
   - VISTA Macro Image AUROC: `0.9305`
   - Gate C: `FAIL`
3. In contrast, `reports/V7_FINAL_RESEARCH_QUALIFICATION.md` reports:
   - VISTA Macro Image AUROC: `0.9520`
   - Gate C: `PASS`
4. The narrative report cannot be reproduced from `scripts/run_v7_qualification.py` or `reports/V7_BENCHMARK_RESULTS.json`.
5. It was manually written with contradictory numbers.

---

## 11. Production Claims Supported by Actual Code

| Production Claim in Docs / Reports | Supported by Code? | Exact Source Reference & Reality |
|---|:---:|---|
| "Identical Training and Inference Scoring Formulation" | **NO** | `vista_vlm/models/vista_model.py` L380–386: switches formula depending on `nominal_memory.is_fitted`. |
| "Zero False Alarms (0.00% FAR)" | **PARTIAL** | On synthetic fixtures, conformal threshold produced 0.00% empirical FAR on normal test images, but Clopper-Pearson 95% upper bound is up to $31.23\%$ due to small $N=8$. |
| "Macro Image AUROC 0.9520" | **NO** | Actual benchmark JSON records `0.9305`. |
| "Gate C PASS" | **NO** | `reports/V7_BENCHMARK_RESULTS.json` Gate C is explicitly marked `"status": "FAIL"`. |
| "Autonomous Coverage Across Categories" | **NO** | Cable: 0.00%, Candle: 0.00%, Bottle: 35.71%. |
| "Strict 4-Partition Zero-Leakage Protocol" | **YES** | `get_v7_loaders()` in `scripts/run_v7_qualification.py` cleanly separates partitions. |
| "Finite-Sample Conformal Guarantees" | **YES** | `vista_vlm/calibration/conformal.py` strictly computes $k = \lceil(n+1)(1-\alpha)\rceil$ and rejects $k > n$. |
| "Fail-Closed Checkpoint Integrity" | **YES** | SHA-256 verification on weights and memory bank patches is enforced. |
| "Non-Safety Advisory Boundary" | **YES** | System strictly disclaims safety functions; advisory disclaimers present in schemas and docs. |

---

## Summary of Action Items for V7.1 Hardening

1. **Part 2:** Implement `compute_image_anomaly_score()` in `vista_vlm/models/vista_model.py` with identical equations during both training and inference. Compare memory-first (Approach A) vs staged optimization (Approach B).
2. **Part 3 & 4:** Remove test-label threshold optimization in baseline and VISTA evaluations. Fit thresholds strictly on independent `val_calibration` / `val_tuning` sets.
3. **Part 5:** Close the Image AUROC gap against baselines using principled score fusion, temperature scaling, and multi-scale feature aggregation.
4. **Part 6:** Improve dense localization IoU and Dice using validation-tuned thresholds and multi-scale loss supervision.
5. **Part 7:** Redesign selective prediction in `vista_vlm/calibration/selective.py` to restore autonomous coverage on Cable and Candle with statistical risk bounds.
6. **Part 8 & 9:** Add artifact corruption tests, dataset acquisition scripts, and maintain blocked status for Gate B until genuine data is provided.
7. **Part 10 & 11:** Implement multi-seed validation (5 seeds) and ensure machine-readable qualification JSON is the sole source of truth for all reports.
8. **Part 12 & 13:** Add container smoke test, fix PyTorch transformer warnings, and update documentation.

