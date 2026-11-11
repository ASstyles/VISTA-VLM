# VISTA-VLM V6.0 System Architecture Specification

**Document ID:** `V6-SPEC-ARCH-001`  
**Version:** 6.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Overview & Core Tenets

VISTA-VLM (**Vision Intelligence through Spatial and Textual Alignment**) V6.0 is an industrial-grade anomaly detection and localization platform engineered around five foundational tenets:
1. **Decoupled Detection & Semantic Attribution**: Dense anomaly detection is driven by multi-scale patch features from a self-supervised visual foundation model (DINOv2), while semantic candidate identification and operator explanations are governed by a vision-language verifier (CLIP).
2. **Reference-Calibrated Spatial Guidance**: Neural convolutional upsampling is grounded by a non-parametric nominal memory bank scaled via fixed training normal distribution parameters ($\mu, \sigma, p_{95}, \text{scale\_range}$), eliminating destructive per-image min-max distortion.
3. **Multi-Scale Spatial-Semantic Harmony**: Intermediate representations from early layers (Block 4) and semantic abstractions from deep layers (Block 8, 12) are fused via global FiLM conditioning and progressive convolutional decoding with Squeeze-and-Excitation (SE) channel attention and lateral skips.
4. **Finite-Sample Conformal Calibration**: Predictions are accompanied by split conformal calibration guarantees with exact finite-sample order statistic rank calculation ($k = \lceil(n+1)(1-\alpha)\rceil$) and regularized Mahalanobis Out-of-Distribution (OOD) distance testing.
5. **Deterministic 9-State Fail-Safe Operation**: Autonomous decisions are committed strictly when conformal certainty is verified. Ambiguous, uncalibrated, or disturbed inputs transition to deterministic safe states (`HOLD`, `REJECT`, `QUARANTINE`).

---

## 2. Component Specifications

### 2.1 Visual Foundation Encoder
- **Backbone**: DINOv2 ViT-S/14 (`dinov2_vits14`) (ViT-B/14 supported as secondary scaled variant).
- **Parameters**: 22,050,432 (Frozen backbone, `requires_grad = False`).
- **Input Dimension**: $B \times 3 \times 518 \times 518$.
- **Patch Size**: $14 \times 14$ pixels.
- **Spatial Grid**: $37 \times 37 = 1,369$ tokens.
- **Intermediate Taps**: Transformer blocks 4, 8, and 12.
- **Output Tensors**:
  - `patch_features`: 3 spatial feature maps $[B, 384, 37, 37]$.
  - `cls_features`: 3 global tokens $[B, 384]$.
  - `global_cls`: Final layer CLS vector $[B, 384]$.

### 2.2 Nominal Coreset Memory Bank & Reference Calibration
- **Coreset Reduction**: Supports Exact Greedy $k$-Center selection ($c_m = \arg\max_i \min_{c \in C} \|x_i - c\|_2$) and Approximate Greedy $k$-Center using Johnson-Lindenstrauss random orthogonal projection ($D \to 64$).
- **Reference Distribution Calibration**: Fixed reference statistics are computed on training normal patches to preserve cross-image absolute distance ranking:
  $$s_{\text{cal\_mem}} = \text{clamp}\left(\frac{D(x) - \text{ref\_min}}{\text{ref\_scale}}, 0.0, 1.0\right)$$
- **Spatial Residual Fusion**:
  $$M_{\text{final}} = (1 - w_{\text{mem}}) \cdot M_{\text{neural}} + w_{\text{mem}} \cdot M_{\text{memory}} \quad (w_{\text{mem}} = 0.65)$$

### 2.3 Image-Level Anomaly Discrimination Head
In V6.0, image-level discrimination combines continuous memory bank distance with high-percentile spatial pooling:
$$S_{\text{image}} = 0.70 \cdot s_{\text{cal\_mem}} + 0.30 \cdot s_{\text{topk\_spatial}}$$
where $s_{\text{topk\_spatial}} = \frac{1}{K} \sum_{k=1}^K \text{TopK}(M_{\text{final}}, K)$ with $K = \max(1, 0.005 \times H \times W)$.

### 2.4 Progressive Segmentation Decoder with SE Attention
- **Lateral Skips**: Direct projections from intermediate layers to restore high-frequency boundary details:
  - Skip 8: $384 \to 128$ channels at $37 \times 37$
  - Skip 4: $384 \to 64$ channels at $74 \times 74$
- **Channel Attention**: Squeeze-and-Excitation (SE) blocks on each convolutional stage adaptively recalibrating channel weights.
- **Stage Resolutions**:
  - Stage 1: $256 \to 128$ channels, upsample to $74 \times 74$
  - Stage 2: $128 \to 64$ channels, upsample to $148 \times 148$
  - Stage 3: $64 \to 32$ channels, upsample to $296 \times 296$
  - Stage 4: $32 \to 16$ channels, upsample to $518 \times 518$
- **Auxiliary Heads**: Intermediate deep supervision heads at $148 \times 148$ and $296 \times 296$ preventing gradient decay.

### 2.5 Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_D \mathcal{L}_{\text{dice}} + \lambda_T \mathcal{L}_{\text{tversky}} + \lambda_B \mathcal{L}_{\text{sobel}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_I \mathcal{L}_{\text{image}}$$
- **Binary Focal Loss**: $\alpha=0.75, \gamma=2.0$ for extreme foreground/background pixel imbalance.
- **Soft Dice Loss**: Stabilizes binary contour convergence.
- **Tversky Asymmetric Loss**: $\alpha=0.7, \beta=0.3$ penalizing false negatives on micro-cracks ($<0.1\%$ image area).
- **Sobel Boundary Loss**: Differentiable edge gradient matching forcing crisp boundary transitions.
