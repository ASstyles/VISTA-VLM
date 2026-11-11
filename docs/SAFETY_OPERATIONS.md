# VISTA-VLM V5.0 Industrial Functional Safety Operations Guide

**Document ID:** `V5-DOC-SAF-001`  
**Version:** 5.0.0  
**Target:** Factory Safety Managers & Line Supervisors  
**Classification:** SAFETY COMPLIANCE & OPERATIONAL GUIDE  

---

## 1. Safety Classification and Operational Mandate

VISTA-VLM V5.0 is an:
> **Advisory Quality Inspection Support Tool (SIL 0 / Non-Safety Related)**

### Regulatory Operating Rule
Under **no circumstances** may VISTA-VLM output signals be wired directly to:
1. Primary machine emergency stop circuits (E-STOP loops).
2. Personnel safeguarding interlocks (light curtains, safety gates, presence mats).
3. Hazardous press or robotic motion actuators without independent safety monitoring.

All human safeguarding must be executed by certified Category 4 / PL e safety controllers.

---

## 2. Physical Factory Line Fail-Closed Interlocking

The physical sorting cell must adhere to the **Fail-Closed Principle**:
- **Default Position**: In the event of total electrical power loss, PLC processor fault, or network disconnect, the pneumatic sorting diverter must drop into the **`REJECT` or `HOLD`** position.
- **Part Release**: A part is only released into the nominal production flow (`ACCEPT`) when a cryptographically verified and calibrated inference response is received confirming operational state `NORMAL`.

```
                    +-----------------------------+
                    | VISTA-VLM API Server        |
                    +--------------+--------------+
                                   | (Digital I/O or Industrial Ethernet)
                                   v
                    +-----------------------------+
                    | Line Automation PLC         |
                    +--------------+--------------+
                                   |
         +-------------------------+-------------------------+
         | State: NORMAL           | State: Any Other State  | Heartbeat Lost (>1000ms)
         v                         v                         v
[Actuate Release Gate]     [Actuate Diverter to]     [Fail-Closed Drop to]
   (Discharge Part)            (REJECT Chute)           (HOLD / Quarantine)
```

---

## 3. Operator Interface (HMI) Visual Annunciation Standards

The factory touchscreen dashboard and stack light tower must display deterministic visual states:

| Operational State | Stack Light Indicator | HMI Display Banner | Audible Annunciation | Line Conveyor Status |
| :--- | :--- | :--- | :--- | :--- |
| **`NORMAL`** | Steady **GREEN** | "PART ACCEPTED - NORMAL" | None | Continuous forward movement |
| **`ANOMALY_DETECTED`** | Steady **RED** | "DEFECT DETECTED - REJECT" | Single short chime | Diverter fires to scrap bin |
| **`UNCERTAIN`** | Steady **YELLOW** | "AMBIGUOUS - MANUAL REVIEW" | Double tone | Diverter routes to review queue |
| **`OOD`** | Flashing **RED / YELLOW** | "OUT OF DISTRIBUTION PART" | Warning buzzer | Conveyor pauses; supervisor check |
| **`IMAGE_QUALITY_FAILURE`**| Flashing **YELLOW** | "IMAGE FAULT: FOCUS / LUX" | Intermittent beep | Pause line; inspect camera lens |
| **`CALIBRATION_INVALID`** | Flashing **RED** | "CALIBRATION ERROR" | Alarm klaxon | Line halt; reload signed model |
| **`COMMUNICATION_FAILURE`**| Steady **RED** + Buzzer | "CAMERA / NETWORK TIMEOUT"| Continuous alarm | Immediate line hold; check GigE |
