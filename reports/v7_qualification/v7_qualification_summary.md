# VISTA-VLM V7.0 — Research Qualification & Reliability Report

**Date:** 2026-11-12T00:00:32Z | **VISTA-VLM Version:** 7.0.0 | **PyTorch:** 2.12.1+cpu
**Split Protocol:** `strict_4_partition_zero_leakage`

## 1. Executive Gate Status Matrix

| Gate | Name | Status | Evidence & Audit Notes |
|---|---|:---:|---|
| **Gate A** | Software Engineering | `PASS` | Automated pytest unit and integration tests passing, clean wheel/sdist packaging, ruff linting clean, TypeScript frontend build certified. |
| **Gate B** | Research Validity | `BLOCKED` | Real external dataset assets (MVTec AD full archive, VisA official AWS tar) were not present in data/real/. Procedural synthetic fixtures isolated. |
| **Gate C** | Detection Quality | `FAIL` | Macro Image AUROC: 0.9305 (CI: [0.8324, 1.0000]), Pixel AUROC: 0.9860, AUPRO: 0.8977, IoU: 0.5649, Dice: 0.7174, FAR: 0.00%. Evaluation protocol: strict 4-partition zero-leakage split. |
| **Gate D** | Industrial Pilot | `BLOCKED` | Requires deployed on-site physical hardware with camera integration, takt-time verification, and supervised shadow-mode validation at partner facility. |
| **Gate E** | Functional Safety | `RESTRICTED_ADVISORY_ONLY` | Non-safety-related advisory inspection software. No SIL (IEC 61508) or PL (ISO 13849-1) certification is claimed or implied. Machine emergency stops must remain independent. |

## 2. Track 1: Multi-Category Macro Benchmark (6 Categories)

| Model | Macro Image AUROC | Macro Pixel AUROC | Macro AUPRO | Macro IoU | Macro Dice | Avg False Alarm |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore** | 0.9670 ± 0.0808 | 0.9918 ± 0.0144 | 0.9220 ± 0.1274 | 0.6767 | 0.8006 | 1.39% |
| **PaDiM** | 0.9800 ± 0.0489 | 0.9890 ± 0.0211 | 0.9008 ± 0.1696 | 0.6822 | 0.8044 | 5.55% |
| **VISTA-VLM V7.0** | 0.9305 ± 0.0935 | 0.9860 ± 0.0239 | 0.8977 ± 0.1703 | 0.5649 | 0.7174 | 0.00% |

### Image-Level Non-Parametric Bootstrap 95% Confidence Intervals (1000 Resamples)
- **Image AUROC:** 0.8780 (95% CI: [0.8214, 0.9229])
- **Image AP:** 0.9189 (95% CI: [0.8748, 0.9522])
- **Resamples:** 1000 | **Total Evaluated Samples:** 162

## 3. Track 4: Few-Normal-Shot Sample Efficiency (MVTec Bottle)

| Few-Shot Regime | PatchCore Image AUROC | PaDiM Image AUROC | VISTA-VLM V7.0 Image AUROC | VISTA Pixel AUROC | VISTA AUPRO |
|---|---:|---:|---:|---:|---:|
| **1-shot** | 1.0000 | 0.5052 | 0.8594 | 0.9909 | 0.9060 |
| **2-shot** | 0.9688 | 0.8542 | 0.9531 | 0.9843 | 0.8757 |
| **4-shot** | 0.9844 | 0.9844 | 0.9948 | 0.9922 | 0.9207 |
| **8-shot** | 1.0000 | 1.0000 | 0.9531 | 0.9899 | 0.9028 |
| **16-shot** | 1.0000 | 1.0000 | 0.8958 | 0.9901 | 0.9326 |

## 4. Scientific Failure Analysis & Robustness

- **False Alarms Mitigation:** Independent split-conformal calibration on held-out val_calibration samples + robust MAD reference scaling + top-k spatial pooling.
- **Residual FAR:** 0.00% (Target: <=5.00%)
- **Tiny Defects (<0.1% Area):** Multi-scale intermediate extraction (DINOv2 Block 4 fine detail skip concatenated with Block 8 contextual tokens) + Tversky asymmetric loss + dynamic defect stratification.
- **OOD Fail-Closed Disposition:** Uncalibrated categories or distance > threshold strictly escalate to HUMAN_REVIEW / QUARANTINE.

---
*Report generated automatically by `scripts/run_v7_qualification.py` with reproducible configuration.*