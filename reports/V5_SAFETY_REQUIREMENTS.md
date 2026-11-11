# VISTA-VLM V5.0 Functional Safety and Operational Requirements Specification

**Document ID:** `V5-REQ-SAF-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  
**Authors:** Functional Safety Engineer & Industrial Automation Engineer  
**Repository Branch:** `v5-real-world-qualification`  

---

## 1. System Safety Requirements (SSR)

### SSR-01: Deterministic Operational State Machine
The system must maintain an explicit, deterministic operational state machine for every inspection query. Ambiguous, intermediate, or uninitialized states are prohibited.

```
                         [Incoming Image]
                                |
                   +------------v------------+
                   | Ingress Integrity Audit |
                   +------------+------------+
                                |
        +-----------------------+-----------------------+
        | Corrupted / Truncated | Focus / Lux Failure   | Valid Bytes
        v                       v                       v
 [COMMUNICATION_FAILURE]   [IMAGE_QUALITY_FAILURE]   [Model & Calibration Check]
     (Disp: HOLD)               (Disp: HOLD)                    |
                                            +-----------+-----------+
                                            | Missing   | Calibrated
                                            v           v
                                   [CALIBRATION_INVALID]  [Feature Extraction]
                                      (Disp: QUARANTINE)        |
                                                    +-----------+-----------+
                                                    | OOD Dist  | In-Dist
                                                    v           v
                                                  [OOD]    [Anomaly Scoring]
                                            (Disp: QUARANTINE)  |
                                    +---------------------------+---------------------------+
                                    | s <= tau_normal           | s in (tau_norm, tau_anom) | s >= tau_anom
                                    v                           v                           v
                                 [NORMAL]                  [UNCERTAIN]             [ANOMALY_DETECTED]
                               (Disp: ACCEPT)          (Disp: HUMAN_REVIEW)          (Disp: REJECT)
```

---

## 2. The 9-State Operational State Matrix

| State Identifier | Trigger Condition | Conformal Validity | Physical Line Disposition | Operator Action Required |
| :--- | :--- | :---: | :---: | :--- |
| **`NORMAL`** | $s(x) \le \tau_{\text{normal}} \land \text{In-Dist} \land \text{Calibrated}$ | Guaranteed | **`ACCEPT`** | None (Part released to next station) |
| **`ANOMALY_DETECTED`** | $s(x) \ge \tau_{\text{anom}} \land \text{In-Dist} \land \text{Calibrated}$ | Guaranteed | **`REJECT`** | Part diverted to scrap / rework bin |
| **`UNCERTAIN`** | $\tau_{\text{normal}} < s(x) < \tau_{\text{anom}}$ (Ambiguous band) | Borderline | **`HUMAN_REVIEW`** | Part diverted to manual visual inspection queue |
| **`OOD`** | Mahalanobis feature distance $D_{\text{Maha}} > \tau_{\text{ood}}$ | N/A | **`QUARANTINE`** | Part quarantined; verify correct lot/model loaded |
| **`IMAGE_QUALITY_FAILURE`** | Defocus blur $\sigma > 1.2$, illumination $< 10\text{ lux}$ | N/A | **`HOLD`** | Clean lens; inspect strobe LED power |
| **`MODEL_UNAVAILABLE`** | Checkpoint hash mismatch or missing weights | N/A | **`HOLD`** | Reload model checkpoint from signed artifact |
| **`CALIBRATION_INVALID`** | Uncalibrated model queried in production pipeline | N/A | **`QUARANTINE`** | Execute split conformal calibration script |
| **`COMMUNICATION_FAILURE`** | Sensor packet loss, network disconnect, timeout | N/A | **`HOLD`** | Inspect camera GigE/USB-3 connection |
| **`HUMAN_REVIEW_REQUIRED`** | Audit sample selected or operator review active | Active | **`HUMAN_REVIEW`** | Quality supervisor review and sign-off |

---

## 3. Physical Hardware Interlocking Requirements (HIR)

- **HIR-01: Fail-Closed Diverter**:
  All physical pneumatic diverters, reject gates, and sorting flippers must be wired fail-closed (normally closed). In the event of total electrical power loss or control signal loss, parts must automatically drop into the `REJECT` or `HOLD` chute.
- **HIR-02: Heartbeat and Watchdog Supervision**:
  The inspection server must publish a hardware heartbeat pulse every $500\text{ ms}$ over digital I/O or industrial Ethernet (EtherNet/IP or PROFINET). If the pulse lapses for $> 1,000\text{ ms}$, the line PLC must trigger an emergency stop (`E-STOP`) or pause line feeding.
- **HIR-03: Sensor Ingress Verification**:
  Every inspection frame must be checked for minimum spatial MTF (Laplacian variance $> 100.0$) and mean intensity ($\mu \in [30, 225]$). Violations must immediately transition the state to `IMAGE_QUALITY_FAILURE`.

---

## 4. Cryptographic Traceability and Audit Logging Requirements (TLR)

- **TLR-01: Immutable Inspection Records**:
  Every inspection event must be logged to an append-only JSON Lines ledger (`/var/log/vista/inspection_audit.jsonl`).
- **TLR-02: Mandatory Audit Log Fields**:
  Each entry must contain:
  ```json
  {
    "inspection_id": "insp_20261111_092348_7482",
    "timestamp_utc": "2026-11-11T13:53:48.129Z",
    "camera_id": "CAM_STATION_04",
    "image_sha256": "4a7f9b2c3d8e...",
    "model_name": "VISTA-VLM",
    "model_version": "5.0.0",
    "checkpoint_sha256": "8f3e2b1a9c...",
    "operational_state": "ANOMALY_DETECTED",
    "disposition": "REJECT",
    "anomaly_score": 0.8412,
    "calibrated_threshold": 0.5000,
    "p_value": 0.0012,
    "ood_distance": 1.42,
    "anomaly_map_sha256": "1d8a4f9e...",
    "execution_time_ms": 649.0,
    "reviewer_id": null
  }
  ```
- **TLR-03: Retention Guarantee**:
  Audit logs must be cryptographically hashed daily and retained for a minimum of **7 years** to support regulatory audits, customer recall investigations, and liability defense.
