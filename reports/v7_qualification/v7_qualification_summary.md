# VISTA-VLM V7.1 — Research Qualification & Reliability Report

**Date:** 2026-11-12T14:55:36Z | **VISTA-VLM Version:** 7.1.0 | **PyTorch:** 2.12.1+cpu
**Split Protocol:** `strict_4_partition_zero_leakage`

## 1. Executive Gate Status Matrix

| Gate | Name | Status | Evidence & Audit Notes |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | Automated pytest unit and integration tests passing, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified. |
| **Gate B** | Research Validity | `BLOCKED` | Real external dataset assets (MVTec AD full archive, VisA official AWS tar) were not present in data/real/. Procedural synthetic fixtures isolated. |
| **Gate C** | Detection Quality | `FAIL` | Macro Image AUROC: 0.9618 (CI: [0.8636, 1.0000]), Pixel AUROC: 0.9874, AUPRO: 0.8938, IoU: 0.5783, Dice: 0.7258, FAR: 0.00%. Evaluation protocol: strict 4-partition zero-leakage split. |
| **Gate D** | Industrial Pilot | `BLOCKED` | Requires deployed on-site physical hardware with camera integration, takt-time verification, and supervised shadow-mode validation at partner facility. |
| **Gate E** | Functional Safety | `RESTRICTED_ADVISORY_ONLY` | Non-safety-related advisory inspection software. No SIL (IEC 61508) or PL (ISO 13849-1) certification is claimed or implied. Machine emergency stops must remain independent. |

## 2. Track 1: Multi-Category Macro Benchmark (6 Categories)

| Model | Macro Image AUROC | Macro Pixel AUROC | Macro AUPRO | Macro IoU | Macro Dice | Avg False Alarm |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore** | 0.9774 ± 0.0553 | 0.9925 ± 0.0129 | 0.9280 ± 0.1113 | 0.6697 | 0.7952 | 16.67% |
| **PaDiM** | 0.9809 ± 0.0468 | 0.9896 ± 0.0195 | 0.9033 ± 0.1668 | 0.6624 | 0.7886 | 16.67% |
| **VISTA-VLM V7.1** | 0.9618 ± 0.0936 | 0.9874 ± 0.0212 | 0.8938 ± 0.1758 | 0.5783 | 0.7258 | 0.00% |

### Image-Level Non-Parametric Bootstrap 95% Confidence Intervals (1000 Resamples)
- **Image AUROC:** 0.9788 (95% CI: [0.9596, 0.9929])
- **Image AP:** 0.9849 (95% CI: [0.9691, 0.9955])
- **Resamples:** 1000 | **Total Evaluated Samples:** 162

## 3. Track 4: Few-Normal-Shot Sample Efficiency (MVTec Bottle)

| Few-Shot Regime | PatchCore Image AUROC | PaDiM Image AUROC | VISTA-VLM V7.1 Image AUROC | VISTA Pixel AUROC | VISTA AUPRO |
|---|---:|---:|---:|---:|---:|
| **1-shot** | 0.9688 | 0.5573 | 0.9948 | 0.9914 | 0.9074 |
| **2-shot** | 0.9896 | 0.8490 | 1.0000 | 0.9908 | 0.9032 |
| **4-shot** | 0.9948 | 1.0000 | 1.0000 | 0.9898 | 0.9098 |
| **8-shot** | 1.0000 | 1.0000 | 1.0000 | 0.9849 | 0.8886 |
| **16-shot** | 1.0000 | 1.0000 | 1.0000 | 0.9882 | 0.9064 |

## 4. Scientific Failure Analysis & Robustness

- **False Alarms Mitigation:** Independent split-conformal calibration on held-out val_calibration samples + robust MAD reference scaling + top-k spatial pooling.
- **Residual FAR:** 0.00% (Target: <=5.00%)
- **Tiny Defects (<0.1% Area):** Multi-scale intermediate extraction (DINOv2 Block 4 fine detail skip concatenated with Block 8 contextual tokens) + Tversky asymmetric loss + dynamic defect stratification.
- **OOD Fail-Closed Disposition:** Uncalibrated categories or distance > threshold strictly escalate to HUMAN_REVIEW / QUARANTINE.

---
*Report generated automatically by `scripts/run_v7_qualification.py` with reproducible configuration.*