# VISTA-VLM V5.0 Production Operations, Drift Monitoring and Continuous Learning Guide

**Document ID:** `V5-DOC-OPS-001`  
**Version:** 5.0.0  
**Target:** MLOps & Production Site Reliability Engineers  
**Classification:** PRODUCTION OPERATIONS MANUAL  

---

## 1. Real-Time Distribution Drift Monitoring

In high-volume manufacturing lines, gradual optical degradation, dust accumulation, or subtle vendor material changes can induce statistical drift in model representations.

### 1.1 Cumulative Score Drift Monitor
The production service tracks a rolling window of the last $W = 500$ inspection scores:
- **Nominal Baseline**: $\mu_0, \sigma_0$ recorded during split conformal calibration.
- **Current Rolling Window**: $\bar{s}_W = \frac{1}{W} \sum_{i=1}^W s_i$.
- **Z-Score Drift Metric**:
  $$Z_{\text{drift}} = \frac{\bar{s}_W - \mu_0}{\sigma_0 / \sqrt{W}}$$
- **Operational Alarm Thresholds**:
  - $|Z_{\text{drift}}| \ge 2.58$ ($p \le 0.01$): **WARNING** — Log distribution shift; notify quality engineer.
  - $|Z_{\text{drift}}| \ge 3.29$ ($p \le 0.001$): **CRITICAL ALARM** — Trigger stack light alert; pause autonomous acceptance.

---

## 2. Continuous Learning and Edge Curation Pipeline

To prevent catastrophic forgetting while improving edge performance, VISTA-VLM incorporates an active-learning curation buffer:

```
[Incoming Inspection]
        |
        v
[Selective Prediction Router]
        |
        +---> [NORMAL / High Conf] -----> [Discard / Ephemeral Log]
        |
        +---> [HUMAN_REVIEW / Edge] ----> [Edge Curation Storage Buffer]
                                                    |
                                                    v
                                         [Operator Labeling UI]
                                                    |
                                      +-------------+-------------+
                                      | Confirmed Normal          | Confirmed Defect
                                      v                           v
                           [Add to Coreset Bank]       [Add to Benchmark Suite]
```

### 2.1 Curation Criteria
1. Any sample triggering operational state `UNCERTAIN` or `HUMAN_REVIEW_REQUIRED`.
2. Any sample where the operator overrides the model disposition.
3. Random uniform sample ($1.0\%$ of normal inspections) for ongoing baseline validation.

---

## 3. Physical Hardware Integration & Optical Calibration

### 3.1 Camera Calibration Procedure
1. Mount a high-precision checkerboard calibration target ($10\text{ mm}$ squares) at the product inspection focal plane.
2. Execute intrinsic calibration to ensure radial distortion $k_1, k_2 \le 10^{-4}$.
3. Measure lens Modulation Transfer Function (MTF): spatial resolution must resolve at least $40\text{ lp/mm}$ at $50\%$ contrast.

### 3.2 Conveyor Encoder Synchronization
- **Triggering Delay**: The photoelectric part-detect sensor pulse must be synchronized with conveyor encoder ticks.
- **Actuator Window**: For a conveyor speed of $v = 1.0\text{ m/s}$ and a distance of $d = 0.5\text{ m}$ between camera and diverter, the total available cycle time before pneumatic actuation is:
  $$t_{\text{actuation}} = \frac{d}{v} = 500\text{ ms}$$
- **Sizing Rule**: Total inference latency + network transit must be $< 450\text{ ms}$ to leave a $50\text{ ms}$ mechanical solenoid response margin.
