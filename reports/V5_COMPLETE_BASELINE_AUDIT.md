# VISTA-VLM v5.0 — Baseline Scientific and Engineering Audit Report

**Audit Authority:** Joint Scientific, Safety, and Engineering Audit Committee  
**Date:** November 2026  
**Audited Branch:** `v4-research-production-upgrade`  
**Working Branch:** `v5-real-world-qualification`  
**Starting Commit:** `e7075c075c359994e5aed302fdcf8e2340382545`  
**Baseline Version:** `4.0.0`  
**Target Version:** `5.0.0`  

---

## 1. Executive Summary & Audit Mandate

The V5.0 Industrial Production and Safety Assurance Program requires a rigorous, zero-fabrication scientific and engineering audit of the VISTA-VLM inspection platform. 

This audit assesses the system against the core principle: **Evidence over claims. Correctness over complexity. Real-world performance over impressive documentation.**

### Key Audit Findings
1. **Procedural Synthetic Fixtures Misrepresented as Industrial Datasets:** Benchmark results in V4 were generated using procedural synthetic image generation (`scripts/prepare_benchmark_data.py`), which synthesizes geometric shapes (ellipses, rectangles, polylines) on solid backgrounds. These were written into directories named `data/processed/mvtec` and `data/processed/visa`, conflating synthetic fixtures with real industrial datasets.
2. **Deficient Spatial Localization (Pixel AUROC ~0.5385, AUPRO ~0.2123):** The VISTA-VLM neural decoder operates by upsampling $37 \times 37$ DINOv2 ViT-S/14 patch tokens to $518 \times 518$ (14x spatial upsampling). While the architectural components (FiLM, SE attention, lateral skips) function mathematically, the model was trained for only 5 epochs on synthetic ellipses without empirical feature grounding, causing its localization performance to collapse to near-random chance compared to PatchCore (Pixel AUROC 0.9626) and PaDiM (Pixel AUROC 0.9841).
3. **Incomplete Evaluation Coverage:** In Track 1, VISTA-VLM was evaluated on only 1 in-domain category (`bottle`). The remaining categories (`cable`, `capsule`, `leather`, `tile`, `candle`) were never trained or evaluated in-domain. Furthermore, VISTA-VLM was absent from the Track 4 Few-Shot benchmark table.
4. **Unsafe Calibration & OOD Fallbacks:** In `conformal.py` and `ood.py`, uncalibrated states or missing category prototypes silently fall back to arbitrary heuristic thresholds (`0.5`), heuristic p-values (`1.0 - score`), or cosine distance comparisons against Mahalanobis thresholds. Under safety-critical guidelines, missing calibration must force an immediate `UNCALIBRATED — HUMAN REVIEW REQUIRED` state.
5. **FastAPI Security Vulnerabilities:** `POST /inspect/batch` accepted client-supplied arbitrary server-side filesystem paths (`image_paths`), creating a critical server-side path traversal / arbitrary file read vulnerability. In addition, file uploads were read entirely into RAM before validating byte limits, creating a denial-of-service vulnerability.
6. **Conflation of Software Testing with Safety Certification:** Passing software unit tests was previously conflated with industrial safety readiness. The platform has never undergone functional safety assessment under IEC 61508 or ISO 13849-1, nor has it been deployed in a physical manufacturing facility.

---

## 2. Full Dependency and Execution Trace

```
[ Raw Image Ingestion ]
         │
         ▼
[ Ingestion & Format Validation ] ──► (Reject non-image, check decompression bombs & dims)
         │
         ▼
[ Preprocessing & Normalization ] ──► (VISTATransform: Resize 518x518, bicubic, ImageNet norm)
         │
         ▼
[ DINOv2 ViT-S/14 Backbone ] ─────► (Extract Patch Tokens & CLS from Blocks 4, 8, 12: [B, 384, 37, 37])
         │
         ├───► [ Global Context ] ──► (CLS tokens -> FiLM affine modulation: gamma, beta)
         │
         ├───► [ Local Features ] ──► (Concat patch tokens -> Linear projection: [B, 256, 37, 37])
         │
         ├───► [ Nominal Memory Guide ] ──► (Nearest-neighbor patch distance to reference memory bank)
         │
         ▼
[ Global-Local Feature Fusion ] ────► (FiLM conditioned local features: f_local + Conv(f_local * (1+g) + b))
         │
         ▼
[ Progressive Segmentation Decoder ] ─► (4 ConvBlock stages with SE attention, lateral skips & memory guide)
         │
         ▼
[ Anomaly Heatmap & Scoring ] ─────► (Pixel Anomaly Map [1, 518, 518] & Hybrid Image Score in [0, 1])
         │
         ├───► [ Conformal Calibration ] ──► (Quantile threshold tau_c on independent normal validation set)
         │
         ├───► [ Mahalanobis OOD Detector ] ─► (Regularized shrinkage distance D_M vs OOD threshold)
         │
         ▼
[ Selective Decision Engine ] ─────► (NORMAL | ANOMALY_DETECTED | UNCERTAIN | OOD | UNCALIBRATED)
         │
         ├───► Autonomous Pass/Reject (Only if high-confidence, in-distribution, and calibrated)
         └───► Escalation to Human Inspector (Mandatory if uncertain, OOD, or uncalibrated)
```

---

## 3. Detailed Component Audit Matrix

| Component | File Path | Audit Severity | Identified Vulnerability / Defect | Remediation Required |
|:---|:---|:---:|:---|:---|
| **Data Preparation** | `scripts/prepare_benchmark_data.py` | **HIGH** | Generates synthetic geometric objects directly into `data/processed/mvtec` and `data/processed/visa`, obscuring provenance. | Relocate generator output to `data/synthetic_fixtures/`. Never write synthetic data to real dataset paths. |
| **Dataset Tests** | `tests/test_dataset.py` | **MEDIUM** | Autouse session fixture generates synthetic data if missing on disk, polluting workspace. | Isolate test datasets to `pytest tmp_path` or explicit synthetic fixture directories. |
| **Real Dataset Support** | `vista_vlm/data/` | **HIGH** | No cryptographic SHA-256 manifest system. No explicit segregation of real vs synthetic data. | Establish `data/real/` and `data/manifests/` with SHA-256 hash checks and metadata. |
| **Spatial Decoder** | `vista_vlm/models/decoder/` | **CRITICAL** | Upsampling $37 \times 37 \to 518 \times 518$ without spatial feature guidance produces blurry, inaccurate localization (Pixel AUROC ~0.5385). | Integrate empirical feature memory residuals and high-resolution patch matching into the localization pipeline. |
| **Training Pipeline** | `scripts/train_and_calibrate.py` | **HIGH** | Never called `model.fit_memory_bank()`. Evaluated only 1 category (`bottle`). No Track 4 few-shot evaluation for VISTA. | Fit nominal memory bank during calibration; expand category evaluation; add VISTA to Track 4 few-shot benchmark. |
| **Conformal Calibration** | `vista_vlm/calibration/conformal.py` | **HIGH** | Falls back to heuristic p-values (`1 - score`) and default threshold `0.5` when calibration is absent. | Raise explicit uncalibrated status; forbid autonomous decisions when uncalibrated. |
| **OOD Detection** | `vista_vlm/calibration/ood.py` | **HIGH** | Falls back to cosine distance against Mahalanobis thresholds or returns `is_ood=False` on unknown categories. | Return `OOD_UNCALIBRATED` and escalate to human review immediately. |
| **Selective Prediction** | `vista_vlm/calibration/selective.py` | **HIGH** | Does not handle corrupted calibration, missing models, or sensor failures. | Implement fail-safe state machine with 9 explicit operational states. |
| **AUPRO Metric** | `vista_vlm/evaluation/metrics.py` | **MEDIUM** | Linear threshold sweep (`linspace(0, 1, 100)`) suffers from quantization errors in narrow score ranges. | Implement adaptive threshold sweeping matching empirical score distributions. |
| **Defect Size Analysis** | `vista_vlm/evaluation/metrics.py` | **MEDIUM** | Lacks defect-size stratification (tiny $<0.1\%$, small $0.1-1\%$, medium $1-5\%$, large $>5\%$). | Implement defect size binning and per-bin recall/localization analysis. |
| **API Path Traversal** | `vista_vlm/api/routes.py` | **CRITICAL** | `POST /inspect/batch` accepted raw filesystem paths (`image_paths`). | Deprecate raw path ingestion; restrict uploads to multipart files or sandboxed IDs. |
| **API DoS Vulnerability** | `vista_vlm/api/routes.py` | **HIGH** | `file.file.read()` read complete file into RAM before checking size limit. | Stream read and enforce maximum payload cap dynamically. Add PIL decompression bomb checks. |
| **CI Guardrails** | `.github/workflows/ci.yml` | **MEDIUM** | CI pipeline ran synthetic generator into real data paths and omitted type/security scans. | Update CI to enforce clean separation and trigger on `v5-real-world-qualification`. |
| **Safety Assurance** | Entire Repository | **CRITICAL** | No formal Hazard Analysis, FMEA, Safety Case, or standards applicability assessment. | Author comprehensive functional safety documentation clarifying non-safety-critical advisory status. |

---

## 4. Root Cause Analysis: Spatial Localization Collapse

VISTA-VLM v4 reported a macro Pixel AUROC of 0.5385 on bottle and 0.4647 cross-category. This poor localization performance is attributed to three primary factors:

1. **Information Bottleneck in ViT Patches:**  
   DINOv2 ViT-S/14 takes a $518 \times 518$ image and tokenizes it into non-overlapping $14 \times 14$ pixel patches, yielding a $37 \times 37$ patch grid ($1,369$ tokens). Each patch token represents a receptive field of $14 \times 14$ pixels. A hairline crack or pinhole of size $4 \times 4$ pixels occupies only a small fraction of a single patch token.

2. **Absence of Memory Guidance in Trained Checkpoints:**  
   While `VISTAAnomalyModel` contained the methods `fit_memory_bank` and `compute_memory_guide`, the training script `train_and_calibrate.py` never invoked `fit_memory_bank`. As a result, `model.memory_bank` remained `None`, disabling the empirical distance guide during inference and forcing the convolutional decoder to infer defect contours from synthetic ellipse training data alone.

3. **Pure Neural Decoding vs. Non-Parametric Feature Distance:**  
   Methods like PatchCore achieve Pixel AUROC >0.96 because they measure the exact Euclidean/cosine distance from each test patch to the nearest nominal training patch in the foundation model's feature space. Neural decoders trained on synthetic anomalies often suffer from distribution shift when encountering real anomalies (e.g., subtle discoloration, glass fractures, or structural wiring defects).

**V5 Solution:**  
Integrate the empirical nominal patch distance residual directly into the localization prediction, combining non-parametric foundation feature matching with progressive SE-attention boundary refinement.

---

## 5. Security & Vulnerability Assessment

### SEC-V5-01: Arbitrary File System Path Ingestion in Batch API
- **Endpoint:** `POST /inspect/batch`
- **Vulnerability:** The request schema `BatchInspectionRequest` took `image_paths: List[str]`. The route checked `if os.path.exists(path): pipeline.inspect(path)`. An unauthenticated client could probe arbitrary server paths (`/etc/shadow`, `C:\Windows\System32\...`), exposing system topology and triggering unauthorized file access.
- **Remediation:** Remove raw path ingestion from public endpoints. Support multipart batch uploads with strict content type validation.

### SEC-V5-02: Unbounded In-Memory File Uploads (Denial of Service)
- **Endpoint:** `POST /inspect`
- **Vulnerability:** `contents = file.file.read()` reads the entire request payload into heap memory before checking `len(contents) > 20 * 1024 * 1024`. A concurrent attack with several 100MB streams could exhaust available RAM.
- **Remediation:** Read chunk-by-chunk with a strict 20MB limit. Terminate connection immediately if the limit is exceeded.

### SEC-V5-03: Decompression Bomb & Image Parsing Exploits
- **Endpoint:** `POST /inspect`
- **Vulnerability:** Validation relied solely on initial magic bytes (`\x89PNG`, `\xff\xd8\xff`). It did not validate image dimensions, aspect ratio, or protect against decompression bombs (`PIL.Image.DecompressionBombError`).
- **Remediation:** Enforce `Image.MAX_IMAGE_PIXELS = 25_000_000`, verify decoded PIL image dimensions, enforce maximum dimensions (4096 x 4096), and catch decompression exceptions cleanly.

---

## 6. Functional Safety Assessment Summary

- **Safety Classification:** VISTA-VLM is strictly an **Advisory Quality Inspection Support Tool**. It is **NOT** a safety-related control system under IEC 61508 or ISO 13849-1.
- **Prohibited Action:** VISTA-VLM must **NEVER** be connected to an autonomous machine safety interlock, emergency stop, or safety-critical product release mechanism without independent, safety-rated hardware safeguards (e.g., safety PLCs, optical light curtains).
- **Fail-Safe Policy:** When uncertainty, sensor failure, or uncalibrated conditions arise, the decision must default to `HOLD / ESCALATE_HUMAN_REVIEW`. It must **NEVER** default to `NORMAL / PASS`.

---

## 7. Audit Conclusion & Next Steps

The V4 baseline has been preserved and audited. All defects and architectural bottlenecks have been identified with zero obfuscation. 

Proceed immediately to **Phase 2 (Real-Data Separation & Dataset Infrastructure)** and **Phase 4 (Localization Architecture Breakthrough)**.

