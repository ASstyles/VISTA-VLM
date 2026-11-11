# VISTA-VLM V6.0 Functional Safety & Industrial Risk Assessment Package

**Document ID:** `V6-DOC-SAF-001`  
**Version:** 6.0.0  
**Classification:** Non-Safety Advisory Quality Inspection Software Assessment  
**Standard Compliance Review:** IEC 61508:2010 | ISO 13849-1:2023 | IEC 62061:2021 | ISO 12100:2010 | EU 2023/1230

---

## 1. Safety Classification & Legal Disclaimer

> [!CAUTION]
> **LEGAL AND FUNCTIONAL MANDATE**  
> **VISTA-VLM is non-safety-related advisory inspection software. No SIL (IEC 61508) or PL (ISO 13849-1) certification is claimed or implied.**  
> VISTA-VLM is designed exclusively for product quality assurance and process optimization. It **MUST NEVER** be used as a safety-related part of a control system (SRP/CS) or as the primary protective measure against machinery hazards. Under no circumstances may VISTA-VLM output signals be wired into machine emergency-stop circuits, light curtain loops, or safety interlocking controllers.

---

## 2. Intended-Use Specification & Operational Design Domain (ODD)

### 2.1 Intended Use
VISTA-VLM is intended solely for automated optical inspection (AOI) to detect, localize, and classify visual surface defects, dimensional non-conformities, and foreign objects on manufactured articles.

### 2.2 Excluded Applications (Misuse Prohibitions)
- Personnel protection or presence sensing near hazardous moving parts.
- E-Stop actuation or hazardous energy isolation.
- Medical device diagnosis or patient intervention.
- Autonomous sorting of explosive, toxic, or hazardous materials without independent physical containment.

### 2.3 Operational Design Domain (ODD) Boundaries
| Parameter | Permissible Operating Range | Action Outside ODD |
|---|---|---|
| **Ambient Illumination** | 800 - 1500 Lux (telecentric lighting) | Trigger `IMAGE_QUALITY_FAILURE` (HOLD) |
| **Motion Velocity** | < 0.5 m/s (strobe synchronized) | Blurry image detected -> Quarantine |
| **Field of View (FoV)** | Product fully centered; minimum 10% margin | Edge occlusion -> Escalate to Operator |
| **Image Resolution** | 518 x 518 nominal (scaled from >=1080p source) | Below 16x16 or above 4096 -> Reject 422 |
| **Ambient Temperature** | 0°C to +45°C (Edge Industrial PC rated) | Hardware thermal warning -> Fail-Closed |

---

## 3. Applicable Regulatory & Standards Matrix

1. **IEC 61508:2010 (Parts 1-7)**: *Functional safety of electrical/electronic/programmable electronic safety-related systems*.
   - VISTA-VLM is classified as an external non-safety system. It interfaces with external industrial PLCs via standard TCP/IP or Fieldbus (Profinet/EtherNet/IP).
2. **ISO 13849-1:2023**: *Safety of machinery — Safety-related parts of control systems*.
   - Physical machine safeguarding must achieve designated Performance Levels (e.g., PL d or PL e) using independent safety relays or safety PLCs.
3. **ISO 12100:2010**: *Safety of machinery — General principles for design — Risk assessment and risk reduction*.
   - Three-step method applied: inherently safe physical diverter design, technical safeguards, and operator annunciation.
4. **EU Machinery Regulation (EU) 2023/1230**:
   - Ensures machine autonomy software components do not circumvent physical risk mitigations.
5. **EU Artificial Intelligence Act (Regulation (EU) 2024/1689)**:
   - High-risk industrial safety categorization avoided by strictly confining AI outputs to advisory quality metadata.

---

## 4. Failure Modes, Effects, and Criticality Analysis (FMEA/FMECA)

| ID | Component | Failure Mode | Root Cause | Local Effect | System Effect | Severity (S) | Occur (O) | Detect (D) | RPN | Risk Mitigation & Safety Interlock |
|---|---|---|---|---|---|:---:|:---:|:---:|:---:|---|
| **FM-01** | Visual Backbone / Memory Bank | False Negative (Defect Escape) | Low-contrast defect (<0.1% area), distribution drift, memory bank undersampling | Anomaly score < threshold | Defective part passed to next manufacturing stage | 7 | 3 | 4 | **84** | Split conformal upper-tail calibration + multi-scale intermediate skips (Block 4+8) + dual human review band |
| **FM-02** | Camera / Lighting | Optical Degradation | Lens contamination, lamp failure, defocus | Blurry or underexposed image | High false alarm or missed defect | 6 | 4 | 2 | **48** | Pre-inference image validation: Laplacian variance blur check + luminance thresholding -> `IMAGE_QUALITY_FAILURE` |
| **FM-03** | Inference Server | Deadlock / Process Crash | Out-of-memory (OOM), memory leak, host kernel panic | API endpoint unreachable | Inspection line halted | 5 | 2 | 1 | **10** | System watchdog + hardware fail-closed diverter drop to HOLD + container restart policy |
| **FM-04** | OOD Detector | False In-Distribution (Semantic Shift) | Novel product variant inspected with wrong category profile | Mahalanobis distance fails to flag anomaly | Inappropriate threshold applied | 8 | 2 | 4 | **64** | Ledoit-Wolf shrinkage covariance + fail-closed rejection on uncalibrated profiles -> `QUARANTINE` |
| **FM-05** | Network / PLC Interface | Latency Spike / Dropped Packet | Switch congestion, packet drop, duplex mismatch | Decision exceeds takt-time deadline | Part travels past diverter gate | 7 | 2 | 2 | **28** | Hardware FIFO buffer + PLC tracking timer: if decision missing by deadline, PLC default-diverts part to HOLD |

---

## 5. Fault Tree Analysis (FTA): Defect Escape Event

```
                             [DEFECT ESCAPE OCCURS]
                                       |
                                     (OR)
                   +-------------------+-------------------+
                   |                                       |
        [VISTA-VLM Issues False ACCEPT]         [Physical Diverter Hardware Jam]
                   |                                       |
                 (AND)                                   (AND)
       +-----------+-----------+               +-----------+-----------+
       |                       |               |                       |
[Defect Score]       [Sample Falls Outside]  [Diverter]             [Fault Sensor]
[<= Threshold]       [Uncertainty Band]      [Mechanism Stuck]      [Fails to Trip]
       |
     (OR)
   +---+---+
   |       |
[Tiny Defect] [Extreme Shift]
[Attenuated]  [False Normal]
```

---

## 6. Goal Structuring Notation (GSN) Safety Argument

- **Goal G1**: VISTA-VLM operations shall never lead to dangerous factory conditions or hazardous machine states.
  - **Context C1**: System deployed strictly as advisory quality inspection tool.
  - **Strategy S1**: Argument by complete separation of machine safety controls and automated quality sorting.
    - **Goal G1.1**: Machine personnel safeguarding is autonomous and independent of AI inspection outputs (backed by PL e safety circuits).
    - **Goal G1.2**: In the event of software failure, communication loss, or uncertainty, physical handling defaults to fail-closed state (`HOLD` / `QUARANTINE`).
    - **Goal G1.3**: Selective prediction engine prevents autonomous `ACCEPT` on out-of-distribution or ambiguous inputs.

---

## 7. Residual Risk Register & Operating Mandates

| Risk ID | Residual Risk Description | Mitigating Technical Control | Operational Requirement |
|---|---|---|---|
| **RR-01** | Micro-defects (<0.05% area) near optical resolution limits | Multi-scale Block 4 lateral skip connections | Periodic manual quality audit of 1% random sample |
| **RR-02** | Novel defect types not covered by training distribution | Upper-tail conformal p-values + Mahalanobis OOD | Mandatory re-calibration when manufacturing retools |
| **RR-03** | PLC communication timeout under heavy network traffic | Hard real-time PLC tracking timer | Dedicated isolated industrial VLAN for inspection subnet |

---

*Verified by Functional Safety & Systems Engineering Team — VISTA-VLM V6.0*
