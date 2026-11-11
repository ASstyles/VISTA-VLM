# VISTA-VLM V5.0 Scientific Research and Evaluation Protocol

**Document ID:** `V5-DOC-RES-001`  
**Version:** 5.0.0  
**Target:** Computer Vision Researchers & Benchmark Authors  
**Classification:** SCIENTIFIC RESEARCH PROTOCOL  

---

## 1. Non-Negotiable Research Principles

All research and experimentation conducted under the VISTA-VLM project must strictly adhere to the five non-negotiable principles:
1. **Principle 1.1: No Fabricated Evidence**: Synthetic procedural fixtures must never be presented as real public or industrial datasets. Missing results must appear as missing or `BLOCKED`, never as invented numbers or copied averages.
2. **Principle 1.2: Strict Separation of Evidence**: Maintain strict physical boundaries between synthetic fixtures (`data/synthetic_fixtures/`), real public research datasets (`data/real/`), and partner factory telemetry.
3. **Principle 1.3: Traceability**: Every benchmark claim must cite its commit hash, hardware details, dataset manifest SHA-256 digest, random seed, and raw output JSON.
4. **Principle 1.4: Honest Completion Status**: Every task must be designated `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or `PARTIAL`.
5. **Principle 1.5: Preserved Baselines & Controls**: Difficult categories or test samples must never be removed or weakened to inflate metric appearances.

---

## 2. Standardized Benchmark Tracks

Research evaluations must be reported across four standardized tracks:

### Track 1: In-Domain Supervised / Unsupervised Anomaly Detection
- Category-specific models fitted/trained exclusively on nominal images of the target category.
- Evaluated against PatchCore and PaDiM under identical input resolutions ($518 \times 518$).

### Track 2: Cross-Category Zero-Shot Transfer
- Source model trained on category A (`bottle`) and evaluated zero-shot on unseen target categories without retraining or threshold adjustment.
- Measures domain-invariant representation learning.

### Track 3: Zero-Shot Vision-Language Benchmark
- Pure prompt-ensemble models (WinCLIP, AnomalyCLIP) evaluated with zero target training images.
- Establishes the performance ceiling of language supervision alone.

### Track 4: Few-Shot Adaptation ($k \in \{1, 2, 4, 8, 16\}$)
- Evaluates sample efficiency when only $k$ normal reference samples are available for memory coreset fitting.

---

## 3. Cryptographic Manifest and Leakage Verification Protocol

Before running any benchmark:
1. Generate the immutable dataset manifest:
   ```bash
   python -c "from vista_vlm.data.manifest_manager import generate_dataset_manifest; generate_dataset_manifest('data/real/mvtec_ad', 'data/manifests/real/mvtec_manifest.json', 'mvtec_ad', '5.0.0')"
   ```
2. Verify that `manifest["is_valid"]` is `True`. The manifest manager enforces:
   - Zero image hash duplicates between train, val, and test splits.
   - 1:1 pixel spatial dimension match between raw images and ground-truth masks.
   - Exact category folder schema conformance.
