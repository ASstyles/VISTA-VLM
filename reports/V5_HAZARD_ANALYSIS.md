# VISTA-VLM V5.0 Hazard Analysis and Risk Assessment (HARA & FMECA)

**Document ID:** `V5-REP-HAZ-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  
**Authors:** Functional Safety Engineer & Quality Assurance and Verification Engineer  
**Repository Branch:** `v5-real-world-qualification`  

---

## 1. System Scope and Operational Safety Classification

VISTA-VLM V5.0 is formally classified under industrial safety standards as an:
> **Advisory and Supervised Quality Inspection Support Tool**  
> *(SIL 0 / Non-Safety Related Software under IEC 61508 / ISO 13849-1)*

### Permissible Operating Boundaries
- **Permitted Function**: Anomaly detection, spatial defect localization, defect sizing, and quality grading for scrap reduction and automated sorting.
- **Prohibited Function**: VISTA-VLM **must NEVER** be deployed as a primary safety-critical interlock or autonomous machine safety guard (e.g., light curtain replacement, emergency stop actuator, or personnel hazard prevention).
- **Physical Line Principle**: The physical sorting mechanism must fail **closed** (defaulting to `REJECT` or `HOLD` upon signal loss or watchdog timeout).

---

## 2. Failure Modes, Effects and Criticality Analysis (FMECA)

Scoring Scale:
- **Severity ($S$)**: 1 (Insignificant) to 10 (Hazardous without warning).
- **Occurrence ($O$)**: 1 (Extremely unlikely) to 10 (Inevitable).
- **Detection ($D$)**: 1 (Certain detection before line progression) to 10 (Undetected escape).
- **Risk Priority Number ($RPN$)**: $S \times O \times D$ (Range: 1 to 1,000). Action mandatory for RPN $\ge 100$ or Severity $\ge 8$.

### FMECA Audit Matrix
| ID | Failure Mode | Cause | End Effect | $S$ | $O$ | $D$ | Initial RPN | Engineered Mitigation Control | Mitigated $S$ | Mitigated $O$ | Mitigated $D$ | Final RPN | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **FM-01** | **False Pass (Defect Escape)** | Low-contrast defect; model threshold too permissive | Compromised product shipped to customer; potential downstream failure | 9 | 3 | 4 | **108** | Split conformal calibration ($\alpha=0.05$); dual-threshold selective prediction routing ambiguous samples to human review | 9 | 1 | 2 | **18** | **PASS** |
| **FM-02** | **Silent Pipeline Timeout** | Processing hang; GPU lockup; network drop | Uninspected part progresses down conveyor unchecked | 9 | 2 | 5 | **90** | Hardware watchdog timer ($500\text{ ms}$ timeout); PLC default diverter set to fail-closed (`HOLD`) | 9 | 1 | 1 | **9** | **PASS** |
| **FM-03** | **Optical / Strobe Failure** | LED bank power supply failure; ambient darkness | Model receives black or dark frames, failing to detect surface cracks | 8 | 3 | 3 | **72** | Ingress luminosity verification; Mahalanobis OOD flag; immediately transition to `IMAGE_QUALITY_FAILURE` | 8 | 1 | 2 | **16** | **PASS** |
| **FM-04** | **Camera Defocus Drift** | Mechanical vibration shifts lens focus ring | Blurred image flattens micro-defects into normal texture | 8 | 2 | 4 | **64** | Spatial Laplacian MTF focus monitor; rejects frames with focus score below threshold | 8 | 1 | 2 | **16** | **PASS** |
| **FM-05** | **False Alarm Overkill Flood** | Uncalibrated transfer; lighting flicker | High false scrap; conveyor line halts; operator bypasses system | 5 | 4 | 2 | **40** | Category memory bank fitting; 99th-percentile scoring; automated drift warning | 5 | 2 | 1 | **10** | **PASS** |
| **FM-06** | **OOD / Foreign Part Input** | Wrong batch loaded; tool left in inspection chamber | Hallucinated prediction or arbitrary decision | 7 | 2 | 3 | **42** | Fitted Mahalanobis OOD detector; immediately flags `OOD` with disposition `QUARANTINE` | 7 | 1 | 2 | **14** | **PASS** |
| **FM-07** | **Process Memory Crash** | Gradual tensor leak across $10^5$ cycles | Inspection server crashes mid-shift | 6 | 2 | 3 | **36** | Process supervisor (systemd / k8s); memory ceiling watchdog; 1-hour leak-free verification | 6 | 1 | 2 | **12** | **PASS** |

---

## 3. Hazard and Operability Study (HAZOP)

| Guide Word | Parameter | Process Deviation | Consequence | Safeguard / Protection |
| :--- | :--- | :--- | :--- | :--- |
| **NO** | Image Data | Camera trigger fires but no image bytes received | Conveyor part advances without inspection | Hardware timeout raises `COMMUNICATION_FAILURE`; PLC diverts part to `HOLD` bin |
| **MORE** | Image Data | Image payload exceeds expected size ($> 20\text{ MB}$) | API server memory exhaustion / DoS | `MAX_IMAGE_BYTES` chunked limit returns HTTP 413 |
| **LATE** | Inference Response | Processing latency exceeds line cycle window ($> 1,000\text{ ms}$) | Part passes reject gate before actuator can trigger | Asynchronous edge queue + hardware PLC sync signal |
| **WRONG** | Part Category | Operator selects "bottle" model for "cable" inspection | Model runs uncalibrated, generating severe false alarms | Embedded part QR code / barcode validation in API schema |
| **CORRUPTED**| Pixel Stream | Sensor bit-flip or corrupted transfer packets | Partial image decoded by PIL | `pil_img.verify()` and format bounds verification |

---

## 4. Fail-Safe Architectural Guarantees

1. **Default Safe State**: In all failure conditions (timeout, corrupt frame, uncalibrated model, network disconnect), the system disposition **must default to `HOLD` or `REJECT`**, never `ACCEPT`.
2. **Watchdog Interlock**: An external physical PLC must poll the `/health` endpoint every $1,000\text{ ms}$. If the heartbeat is lost for two consecutive cycles, the PLC must stop line feeding or divert parts to quarantine.

