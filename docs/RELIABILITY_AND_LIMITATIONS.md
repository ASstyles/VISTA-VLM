# VISTA-VLM v4.0 Reliability, Calibration, and System Limitations

## 1. Statistical Reliability Guarantees

### 1.1 Split Conformal Calibration
VISTA-VLM bounds false-alarm risk using Split Conformal Prediction:
$$\mathbb{P}\left(S(X_{n+1}) \le \tau_c \,\Big|\, Y_{n+1} = 0\right) \ge 1 - \alpha$$
- **Exchangeability Assumption**: The calibration samples $\{X_1, \dots, X_n\}$ and test samples $X_{n+1}$ must be independent and identically distributed (i.i.d.) draws from the nominal distribution.
- **Conformal p-value**: For test anomaly score $s$:
  $$p(s) = \frac{1 + \sum_{i=1}^n \mathbb{I}(s_i \ge s)}{n + 1}$$
  If $p(s) < \alpha$, the null hypothesis of nominality is rejected at significance level $\alpha$.

---

## 2. Out-of-Distribution (OOD) Protection

### 2.1 Shrinkage Mahalanobis Distance
Industrial cameras occasionally capture unexpected objects, empty trays, or altered lighting that does not match the training distribution.
- To prevent unpredictable neural hallucinations, features are evaluated against class prototypes using Ledoit-Wolf regularized shrinkage covariance:
  $$\mathbf{\Sigma}_{\text{shrunk}} = (1 - \lambda) \mathbf{\Sigma} + \lambda \frac{\text{Tr}(\mathbf{\Sigma})}{D} \mathbf{I}$$
- **Shrinkage Guarantee**: In high dimensions ($D=384$) with limited calibration samples ($n < D$), empirical sample covariance is ill-conditioned or singular. Shrinkage guarantees positive-definiteness and a well-behaved matrix condition number.

---

## 3. Dual-Risk Selective Prediction & Sample Size Constraints

### 3.1 Risk Separation
In manufacturing:
- **False Alarm (Type I)** causes unnecessary operator reinspection (financial cost).
- **False Pass (Type II / Defect Escape)** delivers defective parts to customers (catastrophic safety/liability cost).

VISTA-VLM evaluates false-pass risk and false-alarm risk independently across the uncertainty band $[\tau_c - \delta, \tau_c + \delta]$.

### 3.2 The Rule of 300 for 1% Risk Claims
Under binomial testing:
To assert with 95% statistical confidence that a production line achieves $\le 1\%$ false-pass risk, zero defect escapes in a validation set of $N$ defective items requires:
$$(1 - 0.01)^N \le 0.05 \implies N \ge \frac{\ln(0.05)}{\ln(0.99)} \approx 298.07$$
Therefore, any dataset with $N_{\text{def}} < 300$ defective samples cannot statistically prove a sub-1% escape rate at 95% confidence. VISTA-VLM v4 explicitly issues a sample size validity warning whenever selective risk is computed on insufficient test sets.

---

## 4. Known Technical Limitations & Mitigations

1. **Spatial Quantization Limit (14x14 Patches)**:
   - *Limitation*: Patches of size $14 \times 14$ at $518 \times 518$ resolution mean individual tokens span $14 \times 14$ pixels. Microscopic pinholes ($<0.05\%$ image area) produce subtle patch deviations.
   - *Mitigation*: The progressive decoder utilizes lateral skip connections from Block 4 to inject high-frequency spatial gradients, combined with Squeeze-and-Excitation channel gating.
2. **Domain Shift on Stochastic Textures**:
   - *Limitation*: Zero-shot transfer from smooth rigid surfaces (`bottle`) to irregular woven materials (`leather`) produces elevated false alarm rates.
   - *Mitigation*: Always fit category-specific calibration thresholds and nominal memory banks (`fit_memory_bank`) before production deployment on new texture categories.
3. **Severe Optical Variations**:
   - *Limitation*: Sudden changes in aperture, focal plane, or illumination can alter foundation feature representations.
   - *Mitigation*: Re-run split conformal calibration (`scripts/train_and_calibrate.py`) whenever camera optics or line lighting are altered.

