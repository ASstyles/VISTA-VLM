# VISTA-VLM V4 Initial Technical Audit Report
**Vision Intelligence through Spatial and Textual Alignment**
*Principal Research & Systems Engineering Audit*
*Date: November 2026*
*Branch: `v4-research-production-upgrade` | Base Baseline: `f9dcbe0`*

---

## 1. Executive Summary

A comprehensive architectural, mathematical, and algorithmic audit of the **VISTA-VLM** codebase was conducted at the commencement of the **V4 Engineering & Research Upgrade**. The objective of this audit is to identify all critical defects, scientific validity issues, model performance gaps, reliability weaknesses, and production engineering bottlenecks prior to any architectural refactoring.

The audit confirms that while earlier critical bugs (SEC-01 PatchCore score inversion, SEC-02 baseline scalar map expansion, SEC-03/SEC-10 AUPRO threshold truncation, SEC-11 unconstrained synthetic blob, and SEC-12 evaluation track conflation) have been remediated in V3, significant scientific and architectural gaps remain in VISTA-VLM. Specifically:

1. **Model Localization & Generalization Gap**: VISTA-VLM's spatial localization on in-domain industrial tasks (Image AUROC 0.7500, Pixel AUROC 0.5382, AUPRO 0.1984 on `bottle`) lags behind conventional memory-bank baselines (PatchCore: 0.9342 Pixel AUROC / 0.6841 AUPRO; PaDiM: 0.9754 Pixel AUROC / 0.8571 AUPRO). The pure synthetic-anomaly supervision approach fails to generalize to fine real defect morphologies and texture shifts.
2. **OOD Covariance Serialization Defect (SEC-13)**: In `OODDetector`, the inverse covariance matrix $\Sigma^{-1}$ is omitted from the saved JSON checkpoint, causing distance evaluation to silently fall back to cosine distance while comparing against a Mahalanobis threshold, inducing false OOD declarations.
3. **Data Provenance Transparency**: The repository's local `data/processed/` files are procedural synthetic benchmark fixtures conforming to MVTec/VisA formats, used when external multi-gigabyte archives are not downloaded. Dataset provenance must be explicitly declared and tracked across all experimental manifests to adhere to Rule 3.1.
4. **Dual Risk Separation in Calibration**: Conformal prediction guarantees false-alarm bounds on normal samples under exchangeability, but does not control false-pass (defect escape) risk. Dual-criterion risk calibration is required.
5. **Production Readiness Separation**: The backend lacks separate liveness and readiness probes, and inference throughput on CPU (~1.6 FPS) requires caching and inference optimization.

---

## 2. Issue Classification Matrix

| Issue ID | Component / File | Severity | Category | Remediation Plan |
|:---|:---|:---:|:---|:---|
| **SEC-13** | `vista_vlm/calibration/ood.py` | **CRITICAL** | Correctness / Reliability | Serialize and restore $\Sigma^{-1}$ inverse covariance in JSON; normalize distance metrics. |
| **SEC-14** | `vista_vlm/models/vista_model.py` & `decoder/segmentation.py` | **HIGH** | Model Performance Weakness | Integrate multi-scale local patch crop extraction, neighborhood feature comparison, and normal-feature residual guidance. |
| **SEC-15** | `scripts/prepare_benchmark_data.py` & `data/` | **HIGH** | Scientific Validity / Transparency | Explicitly annotate dataset manifests with provenance (`procedural_synthetic_fixture` vs `physical_archive`). |
| **SEC-16** | `vista_vlm/calibration/conformal.py` & `selective.py` | **HIGH** | Reliability / Risk Separation | Implement separate false-alarm control (conformal on normal) and false-pass control (empirical risk on defect validation). |
| **SEC-17** | `vista_vlm/models/language_verifier.py` | **MEDIUM** | Model Performance Weakness | Cache text prompt embeddings; implement multi-prompt ensembling and domain-specific templates. |
| **SEC-18** | `vista_vlm/models/pipeline.py` & `optimization/` | **MEDIUM** | Performance Bottleneck | Implement text embedding caching, array reuse, and float16/TorchScript inference profiling. |
| **SEC-19** | `vista_vlm/api/routes.py` & `api/app.py` | **MEDIUM** | Production Engineering | Add `/health/live` and `/health/ready` endpoints, request timeouts, and structured error responses. |
| **SEC-20** | `docs/` & `reports/` | **LOW** | Documentation Weakness | Produce exhaustive V4 architecture, model card, dataset guide, and reproducibility reports. |

---

## 3. Deep Technical Audit Findings

### Issue SEC-13 [CRITICAL]: OOD Detector Covariance Serialization Defect

- **Affected File**: `vista_vlm/calibration/ood.py` (lines 76–96)
- **Root Cause**:
  In `OODDetector.save()`, the prototype vector $\mu$ and scalar threshold $\tau_{\text{OOD}}$ are serialized, but the regularized inverse covariance matrix `self.cov_inv` is omitted:
  ```python
  data = {
      "metric": self.metric,
      "threshold_percentile": self.threshold_percentile,
      "thresholds": self.thresholds,
      "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
  }
  ```
  Upon `load()`, `self.cov_inv` is empty. In `compute_distance()`:
  ```python
  if self.metric == "mahalanobis" and category in self.cov_inv:
      dist = float(np.sqrt(np.dot(diff @ self.cov_inv[category], diff)))
  else:
      # Fallback to cosine distance
      dist = float(1.0 - np.dot(f_norm, mu_norm))
  ```
  Since `category in self.cov_inv` evaluates to `False`, the method silently calculates cosine distance ($d \in [0, 2]$) but compares it against the Mahalanobis distance threshold (which was calibrated on quadratic forms and may be $0.001$ or $15.0$).
- **Technical Consequence**:
  Normal test images are misclassified as Out-of-Distribution, inappropriately triggering human review escalation on healthy production lines.
- **Remediation**:
  Serialize `self.cov_inv` as nested lists or float arrays in the checkpoint JSON, restore `self.cov_inv` in `load()`, and add a unit test validating round-trip Mahalanobis distance consistency.

---

### Issue SEC-14 [HIGH]: Spatial Defect Localization Gap & Texture Domain Shift

- **Affected Files**: `vista_vlm/models/vista_model.py`, `vista_vlm/models/decoder/segmentation.py`
- **Root Cause**:
  DINOv2 ViT-S/14 patch embeddings operate at a fixed patch resolution ($14 \times 14$ pixels). While multi-layer intermediate extraction (blocks 4, 8, 12) provides semantic hierarchy, all blocks share the same spatial token grid ($37 \times 37$ tokens for $518 \times 518$ input).
  Supervising the segmentation decoder purely with synthetic Perlin or geometric masks forces the decoder to act as a synthetic mask detector rather than an intrinsic anomaly detector. When confronted with real industrial defects (subtle hairline scratches, discoloration, structural breaks), the decoder exhibits diffuse heatmaps and lower AUPRO ($0.1984$ vs PatchCore's $0.6841$).
- **Remediation**:
  1. Integrate multi-scale local crop extraction: augment global feature extraction with multi-crop local patches ($259 \times 259$ or $384 \times 384$ crops) to recover fine sub-patch details.
  2. Implement local neighborhood feature comparison: compute cosine distance residuals between test patch embeddings and nominal reference patch distributions.
  3. Combine synthetic anomaly supervision with feature-space reconstruction/discrepancy loss.

---

### Issue SEC-15 [HIGH]: Dataset Provenance Transparency & Manifest Integrity

- **Affected Files**: `scripts/prepare_benchmark_data.py`, `vista_vlm/data/`
- **Root Cause**:
  To support execution in sandboxed or network-isolated offline environments without requiring 5GB+ external downloads, `prepare_benchmark_data.py` generates procedural benchmark fixtures conforming to official MVTec AD and VisA directory schemas, image dimensions ($518 \times 518$), and defect categories.
- **Technical Consequence**:
  If benchmark results do not explicitly declare dataset provenance, an external observer could confuse procedural synthetic test fixtures with raw physical factory acquisitions (violating Section 3.1).
- **Remediation**:
  1. Add explicit metadata tags in `multi_track_benchmark.json`: `dataset_provenance: "procedural_synthetic_benchmark_fixture"`.
  2. Maintain automated validation tests verifying that datasets conform to the canonical schema.
  3. Support seamless switching to downloaded raw archives when available via `--data_root`.

---

### Issue SEC-16 [HIGH]: False-Alarm vs False-Pass Risk Conflation in Calibration

- **Affected Files**: `vista_vlm/calibration/conformal.py`, `vista_vlm/calibration/selective.py`
- **Root Cause**:
  The split conformal calibrator is fitted on normal validation images to establish a $(1 - \alpha)$ quantile threshold $\tau_c$. This provides a distribution-free upper bound on the **false-alarm rate** ($\text{FPR} \le \alpha$). However, industrial quality inspection is critically sensitive to **false-pass rate** (defect escape, $\text{FNR} = 1 - \text{Recall}$).
- **Technical Consequence**:
  A low false-alarm threshold does not guarantee a bounded defect escape rate. Claiming a "1% risk guarantee" based solely on normal calibration is scientifically invalid.
- **Remediation**:
  Implement dual-criterion calibration:
  1. Conformal false-alarm bound on normal validation set ($\alpha = 0.05$).
  2. Empirical selective risk-coverage analysis on an independent held-out defect validation set, computing the exact empirical false-pass rate and coverage trade-off.

---

### Issue SEC-17 [MEDIUM]: Language Verifier Embedding Redundancy & Grounding Ambiguity

- **Affected File**: `vista_vlm/models/language_verifier.py`
- **Root Cause**:
  For every incoming image with $M$ detected ROIs, text candidate prompts were re-tokenized and re-encoded through CLIP's text transformer. Furthermore, generic prompt templates (e.g., `"a close-up photo of a {category} with an unidentified anomaly"`) can have low cosine margin against subtle defects.
- **Remediation**:
  1. Implement pre-computed, cached text embedding matrices for all supported industrial categories.
  2. Add category-tailored prompt ensembles (e.g. specific to glass, metal, texture).
  3. Enforce strict ambiguity thresholds to prevent unwarranted semantic assertions.

---

### Issue SEC-18 [MEDIUM]: Inference Pipeline Optimization & Latency Bottlenecks

- **Affected File**: `vista_vlm/models/pipeline.py`
- **Root Cause**:
  Inference latency on CPU is currently $619$ ms ($1.62$ FPS). Memory re-allocations during base64 heatmap rendering, redundant text tokenization, and lack of half-precision evaluation create avoidable latency overhead.
- **Remediation**:
  1. Cache static CLIP text features at pipeline initialization.
  2. Optimize image resizing and OpenCV heatmap colormapping.
  3. Implement vectorized batch inference pathways.

---

### Issue SEC-19 [MEDIUM]: Backend Production Health Check Separation

- **Affected Files**: `vista_vlm/api/routes.py`, `vista_vlm/api/app.py`
- **Root Cause**:
  The `/health` endpoint returned `status: "healthy"` without verifying whether model checkpoints, conformal calibration parameters, and OOD prototype matrices are loaded and ready to serve inference.
- **Remediation**:
  Implement Kubernetes-standard `/health/live` (process liveness) and `/health/ready` (model readiness with verification of loaded weights and calibration tables).

---

## 4. End-to-End Data Flow Dependency Map

```
[Ingested Image / File Upload]
             │
             ▼
   [VISTATransform: 518x518 RGB]
             │
      ┌──────┴──────────────────────────────────────┐
      ▼                                             ▼
[DINOv2 ViT-S/14 Backbone]                [Global-Local Feature Bank]
(Frozen multi-layer token extraction:      (Blocks 4, 8, 12 CLS & Patch Tokens)
 Blocks 4, 8, 12: 37x37 tokens)                     │
      │                                             │
      ▼                                             ▼
[Global-Local FiLM Fusion] <────────────────────────┘
      │
      ▼
[Progressive Segmentation Decoder]
  - Lateral Skips (Blocks 4, 8)
  - Deep Auxiliary Supervision (148x148, 296x296)
  - Output: Pixel Anomaly Map [1, 518, 518]
      │
      ├──────────────────────────────────────────────┐
      ▼                                              ▼
[Image-Level Scoring]                     [ROI Extraction & Language Verification]
  - Top-k Pixel Pooling (1%)                - Morphological Contours & Bounding Boxes
  - Global Head Classification              - Frozen CLIP ViT-B/16 Crop Feature Encoding
  - Hybrid Score Formulation                - Cached Text Prompt Similarity & Ranking
      │                                     - Ambiguity Detection (Margin < 0.08)
      ▼                                              │
[Conformal Calibration Check]                        │
  - Calibrated Quantile Threshold tau_c             │
  - Finite-Sample p-value                            │
      │                                              │
      ▼                                              │
[OOD Mahalanobis & Prototypes]                       │
  - Regularized Covariance Distance                  │
      │                                              │
      ▼                                              ▼
[Selective Prediction & Human Review Engine] <───────┘
  - Conformal Uncertainty Band [tau_c - delta, tau_c + delta]
  - OOD Escalation Gating
  - Semantic Ambiguity Gating
      │
      ▼
[Structured Unified Inspection Response (JSON + Base64 Heatmap Overlay)]
```

---

## 5. Remediation Roadmap for V4

1. **Step 1: Fix OOD Covariance Serialization (SEC-13)**.
2. **Step 2: Enhance Model Localization Pathway & Multi-Scale Feature Extraction (SEC-14)**.
3. **Step 3: Dual Risk Separation & Empirical False-Pass Calibration (SEC-16)**.
4. **Step 4: Cache Text Embeddings & Optimize Inference Latency (SEC-17, SEC-18)**.
5. **Step 5: Production Health Checks & API Resilience (SEC-19)**.
6. **Step 6: Retrain and Calibrate V4 Checkpoints Across In-Domain Categories**.
7. **Step 7: Execute Full 4-Track Benchmark Suite with Statistical Validation**.
8. **Step 8: Run Complete Ablation Studies & Failure Error Analysis**.
9. **Step 9: Update React 19 Studio Frontend & Comprehensive Documentation**.

