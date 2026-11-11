# VISTA-VLM V6.0 Industrial Pilot Deployment Runbook

**Document ID:** `V6-DOC-PILOT-001`  
**Version:** 6.0.0  
**Target:** 30-Day Manufacturing Plant Deployment  
**Status:** PILOT READY (Gate D: BLOCKED Pending Partner Site Integration)  
**Classification:** NON-SAFETY ADVISORY / SUPERVISED PILOT  

---

## 1. Pilot Overview & Phased Deployment Strategy

Deploying automated AI inspection directly into an active manufacturing line requires a progressive risk-reduction staging protocol. This 30-day pilot runbook outlines the 4-phase transition from passive observation to selective autonomous sorting.

```
[Day 1-5: Shadow Mode] --------> [Day 6-10: Operator-Assisted]
       |                                     |
       v                                     v
[Day 11-20: Selective Semi-Autonomous] -> [Day 21-30: Production Validation]
```

> [!NOTE]
> Prior to on-site commissioning, plant automation engineers can rehearse the entire 4-phase sequence using the VISTA-VLM industrial inspection line simulator:
> ```bash
> python scripts/simulate_inspection_line.py --category bottle --takt_time 500
> ```

---

## 2. Phase-by-Phase Execution Schedule

### Phase 1: Shadow Mode (Days 1 – 5)
- **Objective**: Verify system stability, camera triggering, network throughput, and latency without interfering with physical production.
- **Physical Line Action**: Sorting diverter is **disabled / held open**. VISTA-VLM receives image triggers and logs predictions to database; zero physical parts are rejected.
- **Success Criteria**:
  - $100\%$ camera trigger capture rate across $> 10,000$ consecutive parts.
  - Zero unhandled software crashes or memory leaks.
  - Mean inference latency $\le 650\text{ ms}$ (CPU) or $\le 30\text{ ms}$ (Jetson/GPU).

### Phase 2: Operator-Assisted Mode (Days 6 – 10)
- **Objective**: Establish baseline operator trust and evaluate human-machine interaction.
- **Physical Line Action**: When VISTA-VLM detects an anomaly ($s \ge \tau_{\text{anom}}$), a visual yellow strobe illuminates and the touchscreen dashboard displays the localized defect mask. The line operator visually confirms the defect before tapping `CONFIRM REJECT` to actuate the diverter.
- **Success Criteria**:
  - Operator confirmation agreement rate $> 95\%$.
  - Measurement of false alarms (overkill) and operator feedback curation.

### Phase 3: Selective Semi-Autonomous Mode (Days 11 – 20)
- **Objective**: Automate clear-cut rejections while protecting against ambiguous edge cases.
- **Physical Line Action**:
  - High-confidence defects ($s \ge \tau_{\text{conf}} + \delta$): **Automatically rejected** by pneumatic diverter.
  - High-confidence nominal parts ($s \le \tau_{\text{conf}} - \delta$): **Automatically accepted**.
  - Ambiguous / Uncertain band ($s \in [\tau_{\text{conf}} - \delta, \tau_{\text{conf}} + \delta]$): Diverted to manual review station (`disposition: HUMAN_REVIEW`).
- **Success Criteria**:
  - Human review escalation rate $\le 8.0\%$ of total production volume.
  - Zero confirmed defect escapes passing through the line.

### Phase 4: Production Validation (Days 21 – 30)
- **Objective**: Full autonomous line qualification and formal customer sign-off.
- **Physical Line Action**: Continuous autonomous operation with active drift monitoring and daily statistical calibration audits.
- **Success Criteria**:
  - System availability / uptime $\ge 99.8\%$.
  - False alarm rate (overkill) $\le 1.0\%$.
  - Defect escape rate $< 0.1\%$ across at least 300 physically audited defective parts with Clopper-Pearson exact one-sided binomial upper confidence bound.

---

## 3. Physical Pre-Installation Checklist

- [ ] **Optics**: Telecentric lens mounted; working distance calibrated to $350\text{ mm} \pm 2\text{ mm}$.
- [ ] **Illumination**: Constant-current diffuse dome LED strobe set to $2,500\text{ lux} \pm 50\text{ lux}$.
- [ ] **Triggering**: Photoelectric retroreflective part sensor wired to Opto-isolated digital input.
- [ ] **Actuation**: Pneumatic blow-off solenoid wired to fail-closed reject chute (100 PSI supply).
- [ ] **PLC Interlock**: Independent Category 4 / PL e safety controllers manage all personnel safeguarding; AI inspection is non-safety advisory.
- [ ] **Computing**: Industrial PC (Intel Core i7, 32GB RAM, Ubuntu 22.04 LTS / Debian 12) running hardened VISTA-VLM container.
