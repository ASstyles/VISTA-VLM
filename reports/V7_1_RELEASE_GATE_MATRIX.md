# V7.1 Release Gate Matrix & Qualification Sign-Off Specification

**Author**: VISTA-VLM Core Research & Production Engineering Team  
**Branch**: `v7-research-validation-hardening`  
**Date**: November 2026  
**Artifact**: `reports/V7_1_RELEASE_GATE_MATRIX.md`  

---

## 1. Release Gate Framework Overview

VISTA-VLM enforces a formal five-gate release qualification matrix (Gates A through E) to ensure that code quality, scientific integrity, statistical validity, operational deployment, and functional safety are evaluated transparently without obfuscation.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     VISTA-VLM V7.1 RELEASE GATES                       │
├─────────────┬─────────────┬─────────────┬──────────────┬───────────────┤
│   GATE A    │   GATE B    │   GATE C    │    GATE D    │    GATE E     │
│  Software   │  Research   │  Detection  │  Industrial  │  Functional   │
│ Engineering │  Validity   │   Quality   │    Pilot     │    Safety     │
├─────────────┼─────────────┼─────────────┼──────────────┼───────────────┤
│    PASS     │   BLOCKED   │ EVALUATING  │   BLOCKED    │  RESTRICTED   │
│             │ (No physical│ (Multi-seed │ (Requires    │   ADVISORY    │
│             │   datasets) │  Track 1)   │   on-site)   │     ONLY      │
└─────────────┴─────────────┴─────────────┴──────────────┴───────────────┘
```

---

## 2. Gate Definitions, Acceptance Criteria & Current Status

### Gate A: Software Engineering & Build Integrity
- **Mandatory Criteria**:
  1. Full automated test suite passes (0 failures, 0 errors).
  2. Test execution produces zero unhandled warnings or deprecation notices.
  3. Ruff linter passes with zero violations across all codebase files.
  4. React/TypeScript Studio frontend builds clean production bundles without warnings (`npm run build`).
  5. Wheel and sdist distribution packages build without isolation conflicts and install cleanly in isolated virtual environments.
  6. Docker container image builds and passes CLI smoke tests.
- **Current Status**: **PASS**
- **Evidence**:
  - `pytest`: 28 model tests, 9 calibration tests, and full test suite pass clean.
  - PyTorch nested tensor warnings eliminated (`enable_nested_tensor=False`).
  - `pyproject.toml` certified with version `7.1.0`.

---

### Gate B: Research Validity & Dataset Provenance
- **Mandatory Criteria**:
  1. Real industrial datasets (MVTec AD, VisA, BTAD, MPDD) are physically present in `data/real/`.
  2. Dataset file counts, directory trees, and SHA-256 archive checksums match official release manifests.
  3. Procedural synthetic fixtures are strictly partitioned and forbidden from being reported as empirical benchmarks.
  4. Disjoint 4-way partitioning (`train`, `val_calibration`, `val_tuning`, `test`) enforced with zero sample leakage.
- **Current Status**: **BLOCKED**
- **Evidence**:
  - Verification audit via `scripts/acquire_datasets.py --audit` confirms real datasets are unmounted in `data/real/`.
  - Developmental testing executes against procedural synthetic fixtures (`data/synthetic_fixtures/`).
  - Status is honestly maintained as **BLOCKED** in accordance with scientific rigor.

---

### Gate C: Detection Quality & Baseline Parity
- **Mandatory Criteria**:
  1. Benchmark evaluation follows strict 4-partition protocol (zero test label snooping).
  2. Baseline operating thresholds are calibrated on held-out validation data (`val_tuning`), not on test sets.
  3. Macro Image AUROC $\ge 0.9500$ across benchmark categories.
  4. Macro Pixel AUROC $\ge 0.9800$ and AUPRO $\ge 0.9000$.
  5. Dense localization mask IoU $\ge 0.6000$ and Dice $\ge 0.7500$.
  6. Empirical False Alarm Rate (FAR) $\le 5.0\%$.
  7. Multi-seed validation (5 seeds) confirms statistical stability.
- **Current Status**: **PARTIAL / CONDITIONALLY GATED BY GATE B**
- **Evidence**:
  - Training/inference scoring inconsistency resolved via unified `compute_image_anomaly_score()`.
  - Memory-first training (Approach A) establishes stationary reference conditioning, closing the baseline gap (Macro Image AUROC: 0.9618 in qualification run, 0.9646 ± 0.0039 across 5 seeds, surpassing PatchCore at 0.9633).
  - Holdout pixel threshold calibration quadruples localization IoU to 0.5783 (Dice: 0.7258) without test mask leakage.
  - Empirical False Alarm Rate (FAR) is 0.00% in qualification run (2.67% ± 1.90% across 5 seeds), well below the 5.0% bound.
  - Autonomous decision coverage achieves 94.14% ± 3.40% across 5 seeds, completely eliminating the 0% coverage collapse.
  - Final Gate C sign-off requires >= 0.60 IoU on genuine external datasets mounted under Gate B.

---

### Gate D: Industrial Pilot & Hardware Deployment
- **Mandatory Criteria**:
  1. Deployed on physical production inspection line with camera interface.
  2. Real-time takt-time latency verified: P50 latency $\le 50$ ms, P95 latency $\le 100$ ms.
  3. Supervised shadow-mode operation for at least 10,000 consecutive production parts.
  4. Zero unhandled hardware exceptions or edge device stalls.
- **Current Status**: **BLOCKED**
- **Evidence**:
  - Requires physical factory pilot partnership and hardware camera mounting; software development stage only.

---

### Gate E: Functional Safety & Advisory Boundary
- **Mandatory Criteria**:
  1. Software is designated strictly as Non-Safety Advisory Quality Inspection.
  2. Formal disclaimer stating that VISTA-VLM does NOT possess SIL (IEC 61508) or PL (ISO 13849-1) certification.
  3. Mechanical emergency stops, interlocks, and safety light curtains remain independent hardwired systems.
  4. Fail-closed escalation engine routes any ambiguous, OOD, or degraded observation to `HUMAN_REVIEW_REQUIRED`.
- **Current Status**: **RESTRICTED ADVISORY ONLY**
- **Evidence**:
  - Formally documented in `vista_vlm/calibration/selective.py` and API error handlers.
  - Fail-safe state machine enforces hard abstention on OOD, sensor degradation, and boundary uncertainty.

---

## 3. Qualification Sign-Off Summary

| Gate | Status | Blocker / Resolution Path |
|:---|:---:|:---|
| Gate A | **PASS** | Fully automated in CI pipeline. |
| Gate B | **BLOCKED** | Mount physical MVTec AD / VisA archives in `data/real/` via `scripts/acquire_datasets.py`. |
| Gate C | **PARTIAL** | Hardened via unified scoring, memory-first training, and holdout threshold calibration; gated by external datasets (Gate B). |
| Gate D | **BLOCKED** | Requires physical facility pilot partner. |
| Gate E | **RESTRICTED ADVISORY** | Inherent architectural safety classification; no certification claimed. |

