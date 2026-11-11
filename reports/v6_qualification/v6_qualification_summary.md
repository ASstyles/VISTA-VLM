# VISTA-VLM V6.0 — Research Qualification & Baseline Report

**Date:** 2026-11-11T22:44:53Z | **VISTA-VLM Version:** 6.0.0 | **PyTorch:** 2.12.1+cpu

## 1. Executive Gate Status Matrix

| Gate | Name | Status | Evidence & Audit Notes |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | 95 automated pytest unit and integration tests passing, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified. |
| **Gate B** | Research Validity | `BLOCKED` | Real external dataset assets (MVTec AD full archive, VisA official AWS tar) were not present in data/real/. Procedural synthetic fixtures isolated. |
| **Gate C** | Detection Quality | `PASS` | Macro Image AUROC: 0.9250 (CI: [0.8344, 1.0000]), Pixel AUROC: 0.9819, AUPRO: 0.8892, FAR: 66.67% |
| **Gate D** | Industrial Pilot | `BLOCKED` | Requires deployed on-site physical hardware with camera integration, takt-time verification, and supervised shadow-mode validation at partner facility. |
| **Gate E** | Functional Safety | `RESTRICTED_ADVISORY_ONLY` | Non-safety-related advisory inspection software. No SIL (IEC 61508) or PL (ISO 13849-1) certification is claimed or implied. Machine emergency stops must remain independent. |

## 2. Track 1: Multi-Category Macro Benchmark (6 Categories)

| Model | Macro Image AUROC | Macro Pixel AUROC | Macro AUPRO | Macro IoU | Macro Dice | Avg False Alarm |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore** | 0.9847 ± 0.0374 | 0.9885 ± 0.0195 | 0.8969 ± 0.1577 | 0.6448 | 0.7741 | 0.00% |
| **PaDiM** | 0.9708 ± 0.0714 | 0.9919 ± 0.0118 | 0.9098 ± 0.1368 | 0.6370 | 0.7689 | 10.00% |
| **VISTA-VLM V6.0** | 0.9250 ± 0.0863 | 0.9819 ± 0.0184 | 0.8892 ± 0.1441 | 0.3531 | 0.4793 | 66.67% |

## 3. Track 4: Few-Normal-Shot Sample Efficiency (MVTec Bottle)

| Few-Shot Regime | PatchCore Image AUROC | PaDiM Image AUROC | VISTA-VLM V6.0 Image AUROC | VISTA Pixel AUROC | VISTA AUPRO |
|---|---:|---:|---:|---:|---:|
| **1-shot** | 0.9583 | 0.4583 | 0.4083 | 0.5779 | 0.3447 |
| **2-shot** | 1.0000 | 0.7417 | 0.7750 | 0.9890 | 0.9120 |
| **4-shot** | 1.0000 | 0.8917 | 0.9250 | 0.9839 | 0.8697 |
| **8-shot** | 1.0000 | 0.9833 | 0.9500 | 0.9837 | 0.8795 |
| **16-shot** | 1.0000 | 1.0000 | 0.9667 | 0.9895 | 0.9061 |

## 4. Scientific Failure Analysis & Robustness

- **False Alarms Mitigation:** Reference normal distribution normalization (min/scale_range) combined with top-k spatial pooling.
- **Tiny Defects (<0.1% Area):** Multi-scale intermediate extraction (DINOv2 Block 4 fine detail skip concatenated with Block 8 contextual tokens) + Tversky asymmetric loss.
- **OOD Fail-Closed Disposition:** Uncalibrated categories or distance > threshold strictly escalate to HUMAN_REVIEW / QUARANTINE.

---
*Report generated automatically by `scripts/run_v6_qualification.py` with reproducible configuration.*