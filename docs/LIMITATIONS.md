# VISTA-VLM V6.0 Known Technical Limitations and Operating Constraints

**Document ID:** `V6-DOC-LIM-001`  
**Version:** 6.0.0  
**Status:** APPROVED  
**Target:** Deployers, Automation Architects, and Quality Engineers  

---

## 1. Safety and Regulatory Boundaries

1. **Non-Safety-Related Advisory Inspection Software**:
   VISTA-VLM V6.0 is classified strictly as **non-safety-related advisory inspection software**. No Safety Integrity Level (SIL under IEC 61508) or Performance Level (PL under ISO 13849-1) certification is claimed or implied. It is **prohibited** from being deployed as a primary safety component for human hazard protection, press interlocking, or robotic cell safety perimeter enforcement.
2. **Mandatory Human Escalation**:
   The system must not be configured to auto-pass ambiguous samples (`disposition: HUMAN_REVIEW`). Overriding the selective prediction uncertainty band violates operational design requirements.
3. **Statistical Proof of Escape Rates**:
   A zero-defect test on $< 300$ samples does not mathematically certify $< 1\%$ defect escape rate. Under the Rule of 300, at least **300 physically distinct, verified defective units** must be evaluated without escapes before making high-reliability claims with Clopper-Pearson exact one-sided bounds.

---

## 2. Environmental Breakdown Limits

VISTA-VLM V5.0 predictions cease to be statistically reliable outside the following verified operating windows:

| Environmental Dimension | Reliable Operational Window | Breakdown Threshold | Consequence of Violation |
| :--- | :--- | :--- | :--- |
| **Illumination Variance** | Nominal $\pm 10\%$ intensity | $\pm 20\%$ intensity | Elevated false alarms on reflective surfaces |
| **Optical Defocus** | Gaussian blur $\sigma \le 0.8\text{ px}$ | $\sigma \ge 1.5\text{ px}$ | Small pinholes and micro-cracks missed |
| **Sensor Noise** | Gaussian noise $\sigma \le 0.01$ | $\sigma \ge 0.05$ | Random spatial activation spikes |
| **Part Translation** | Centered within $\pm 10\text{ px}$ | Translation $> 25\text{ px}$ | Edge clipping and spurious perimeter alarms |
| **Part Rotation** | Aligned within $\pm 3^\circ$ | Rotation $> 5^\circ$ | Orientation mismatch against coreset memory bank |
| **Image Compression** | Lossless (PNG) or JPEG Q $\ge 85$ | JPEG Q $< 75$ | Discrete cosine transform (DCT) block artifacts |

---

## 3. Algorithmic and Domain Limitations

1. **Object-to-Texture Zero-Shot Transfer**:
   While transferring between rigid objects (`bottle` $\rightarrow$ `capsule`) achieves $>0.97$ Pixel AUROC, transferring from an object to continuous stochastic textures (`bottle` $\rightarrow$ `leather`) causes elevated false alarms unless at least 1 normal image is provided for reference memory fitting.
2. **Defects Smaller Than Patch Receptive Field**:
   At $518 \times 518$ input resolution, each DINOv2 patch spans $14 \times 14$ pixels ($\approx 2.7\text{ mm}$ for typical setups). While the progressive SE decoder recovers $>83\%$ of sub-patch defects, defects smaller than 4 contiguous pixels ($< 0.001\%$ of image area) may fall below detection thresholds.
3. **Hardware Latency on Standard CPU**:
   FP32 CPU latency on modern x86 hardware is $\approx 649\text{ ms}$ ($\approx 1.5\text{ FPS}$). Deployments requiring cycle times $< 250\text{ ms}$ must utilize GPU acceleration (TensorRT) or OpenVINO INT8 quantization.

