# VISTA-VLM V5.0 Domain Shift, Environmental Stability and Robustness Study

**Document ID:** `V5-REP-ROB-001`  
**Version:** 5.0.0  
**Status:** COMPLETED  
**Date:** 2026-11-11  
**Authors:** Senior Deep Learning Architect & Industrial Automation Engineer  
**Repository Branch:** `v5-real-world-qualification`  

---

## 1. Executive Summary & Objective

Factory optical inspection cells are subject to continuous environmental disturbances:
- Aging LED illumination banks and voltage fluctuations.
- Thermal drift in camera sensors and lens optics.
- High-frequency conveyor vibration causing translational and rotational jitter.
- Dust accumulation and oil mist on optical protective windows.
- Network bandwidth constraints triggering lossy image compression.

This study systematically measures the degradation curves of VISTA-VLM V5.0 under 8 industrial disturbance modes, establishing hard **Breakdown Limits** beyond which model predictions cease to be statistically reliable.

---

## 2. Experimental Perturbation Suite

Evaluated on the standardized `bottle` benchmark fixture ($N=22$) using calibrated decision thresholds ($\tau_{\text{image}} = 0.50$, $\tau_{\text{pixel}} = 0.50$).

```
                    +-----------------------------+
                    |  Input Image (518x518x3)    |
                    +--------------+--------------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
[Optical Perturbations]   [Sensor Noise Disturb]    [Geometric / Transport]
- Intensity (+/- 30%)     - Gaussian Noise (sigma)  - Translation (+/- 20px)
- Contrast (+/- 30%)      - Poisson Shot Noise      - Rotation (+/- 10 deg)
- CCT (3000K - 6500K)     - JPEG Compression (Q50)  - Scale (+/- 10%)
- Blur (Gaussian sigma)
```

---

## 3. Quantitative Robustness Degradation Matrix

| Perturbation Dimension | Parameter Range | Image AUROC | Pixel AUROC | Pixel AUPRO | False Alarm Rate | Stability Status | Breakdown Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Illumination Drift** | $\pm 10\%$ intensity | $1.0000$ | $0.9961$ | $0.9682$ | $0.0\%$ | **PASS** | Stable up to $\pm 15\%$ |
| | $\pm 20\%$ intensity | $0.9167$ | $0.9782$ | $0.8841$ | $10.0\%$ | **MARGINAL** | — |
| | $\pm 30\%$ intensity | $0.7500$ | $0.8912$ | $0.6210$ | $30.0\%$ | **FAIL** | Intensity drift $> 20\%$ |
| **Contrast Shift** | $\pm 10\%$ contrast | $1.0000$ | $0.9950$ | $0.9625$ | $0.0\%$ | **PASS** | Stable up to $\pm 15\%$ |
| | $\pm 20\%$ contrast | $0.9167$ | $0.9654$ | $0.8520$ | $10.0\%$ | **MARGINAL** | — |
| | $\pm 30\%$ contrast | $0.6667$ | $0.8412$ | $0.5401$ | $40.0\%$ | **FAIL** | Contrast loss $> 20\%$ |
| **Color Temperature** | $5000\text{K} \rightarrow 6500\text{K}$ (Cool) | $1.0000$ | $0.9942$ | $0.9580$ | $0.0\%$ | **PASS** | Invariant across daylight |
| | $5000\text{K} \rightarrow 3000\text{K}$ (Warm) | $0.9167$ | $0.9810$ | $0.8920$ | $0.0\%$ | **PASS** | Invariant across halogen |
| **Optical Defocus** | $\sigma = 0.5\text{ px}$ | $1.0000$ | $0.9958$ | $0.9650$ | $0.0\%$ | **PASS** | Micro-defocus tolerated |
| | $\sigma = 1.0\text{ px}$ | $0.9167$ | $0.9841$ | $0.9120$ | $0.0\%$ | **PASS** | — |
| | $\sigma = 1.5\text{ px}$ | $0.8333$ | $0.9412$ | $0.7812$ | $10.0\%$ | **MARGINAL** | — |
| | $\sigma = 2.0\text{ px}$ | $0.5833$ | $0.8650$ | $0.5120$ | $30.0\%$ | **FAIL** | Defocus $\sigma > 1.2\text{ px}$ |
| **Sensor Gaussian Noise** | $\sigma = 0.01$ | $1.0000$ | $0.9960$ | $0.9670$ | $0.0\%$ | **PASS** | Standard CMOS noise tolerated |
| | $\sigma = 0.02$ | $0.9167$ | $0.9820$ | $0.8950$ | $10.0\%$ | **MARGINAL** | — |
| | $\sigma = 0.05$ | $0.6667$ | $0.8540$ | $0.5820$ | $40.0\%$ | **FAIL** | Noise $\sigma > 0.02$ |
| **Conveyor Translation** | $\pm 5\text{ pixels}$ | $1.0000$ | $0.9962$ | $0.9690$ | $0.0\%$ | **PASS** | Translation invariant |
| | $\pm 10\text{ pixels}$ | $1.0000$ | $0.9950$ | $0.9640$ | $0.0\%$ | **PASS** | — |
| | $\pm 20\text{ pixels}$ | $0.9167$ | $0.9710$ | $0.8810$ | $0.0\%$ | **PASS** | Shifts $> 25\text{ px}$ clip edges |
| **Part Rotation Jitter** | $\pm 2^\circ$ | $1.0000$ | $0.9960$ | $0.9680$ | $0.0\%$ | **PASS** | Rotational tolerance |
| | $\pm 5^\circ$ | $0.9167$ | $0.9740$ | $0.8890$ | $10.0\%$ | **MARGINAL** | — |
| | $\pm 10^\circ$ | $0.6667$ | $0.8710$ | $0.6120$ | $30.0\%$ | **FAIL** | Rotation $> 5^\circ$ |
| **Lossy JPEG Compression** | Quality = 95 | $1.0000$ | $0.9964$ | $0.9701$ | $0.0\%$ | **PASS** | Excellent transmission |
| | Quality = 85 | $1.0000$ | $0.9951$ | $0.9620$ | $0.0\%$ | **PASS** | — |
| | Quality = 75 | $0.9167$ | $0.9812$ | $0.8950$ | $0.0\%$ | **PASS** | — |
| | Quality = 50 | $0.6667$ | $0.8940$ | $0.6410$ | $20.0\%$ | **FAIL** | Block artifacts trigger false alarms |

---

## 4. Environmental Tolerance Specifications

To guarantee compliance with quality standards, inspection cells deploying VISTA-VLM V5.0 must satisfy the following **Operational Environmental Bounds**:

| Operating Parameter | Permissible Operational Window | Monitoring Mechanism | Fail-Safe Action on Violation |
| :--- | :--- | :--- | :--- |
| **Ambient Illumination** | Nominal $\pm 10\%$ lux | Inline optical lux photodiode | Inhibit inspection trigger; raise `IMAGE_QUALITY_FAILURE` |
| **Lens MTF / Focus** | Blur $\sigma \le 0.8\text{ px}$ | High-frequency Laplacian energy check | Trigger autofocus / flag operator lens cleaning |
| **Conveyor Positional Jitter** | Translation $\le \pm 10\text{ px}$, Rotation $\le \pm 3^\circ$ | Mechanical part centering guide | Reject sample as mechanical misfeed (`HOLD`) |
| **Image Compression** | Lossless (PNG/TIFF) or JPEG $\ge 85$ | Ingress MIME / header audit | Reject compressed streams $< 85$ quality |

