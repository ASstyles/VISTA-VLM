# VISTA-VLM V5.0 System Architecture Specification

**Document ID:** `V5-SPEC-ARCH-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Overview & Core Tenets

VISTA-VLM (**Vision Intelligence through Spatial and Textual Alignment**) is designed around five foundational engineering tenets:
1. **Separation of Detection from Semantic Classification**: Dense defect localization is driven by intermediate patch features from a self-supervised visual foundation model (DINOv2), while semantic candidate identification and operator explanations are governed by a multimodal vision-language model (CLIP).
2. **Memory-Guided Spatial Residual Architecture**: Neural convolutional upsampling is grounded by a non-parametric nominal memory bank, resolving the localization collapse and achieving state-of-the-art pixel AUROC ($0.9966$) and tiny defect recall ($83.64\%$).
3. **Multi-Scale Spatial-Semantic Harmony**: Dense spatial representations from early layers (Block 4) and semantic abstractions from deep layers (Block 8, 12) are fused via global FiLM conditioning and progressive convolutional decoding with Squeeze-and-Excitation (SE) channel attention.
4. **Statistical Reliability & Conformal Calibration**: Predictions are accompanied by split conformal calibration guarantees ($\alpha = 0.05$) and regularized Mahalanobis Out-of-Distribution (OOD) distance testing. Uncalibrated fallbacks are strictly prohibited.
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

### 2.2 Nominal Coreset Memory Bank & Spatial Residual Fusion
- **Coreset Reduction**: Greedy $k$-center coreset sampling reduces thousands of nominal patch tokens down to $M=1,500$ nominal prototypes.
- **Distance Metric**: Euclidean nearest-neighbor distance computed against normalized patch embeddings:
  $$D(z_i) = \min_{m \in \mathcal{M}} \| z_i - m \|_2$$
- **Spatial Residual Fusion**:
  $$M_{\text{final}} = (1 - w_{\text{mem}}) \cdot M_{\text{neural}} + w_{\text{mem}} \cdot M_{\text{memory}} \quad (w_{\text{mem}} = 0.5)$$
  Non-parametric memory distance grounds spatial localization, while the progressive SE decoder provides boundary refinement and sub-patch noise suppression.

### 2.3 Global-Local Feature Fusion (FiLM Modulation)
- **FiLM Modulation (Default)**: Modulates concatenated local patch features ($3 \times 384 = 1,152$ channels) with affine parameters $[\boldsymbol{\gamma}, \boldsymbol{\beta}]$ computed from global CLS tokens.
  $$\mathbf{F}_{\text{fused}} = \mathbf{F}_{\text{local}} + \text{Conv}_{3\times3}\left((1 + \boldsymbol{\gamma}) \odot \mathbf{F}_{\text{local}} + \boldsymbol{\beta}\right)$$

### 2.4 Progressive Segmentation Decoder with SE Attention
- **Lateral Skips**: Direct projections from intermediate layers to restore high-frequency boundary details:
  - Skip 8: $384 \to 128$ channels at $37 \times 37$
  - Skip 4: $384 \to 64$ channels at $74 \times 74$
- **Channel Attention**: Squeeze-and-Excitation (SE) blocks on each convolutional block adaptively recalibrating channel weights.
- **Stage Resolutions**:
  - Stage 1: $256 \to 128$ channels, upsample to $74 \times 74$
  - Stage 2: $128 \to 64$ channels, upsample to $148 \times 148$
  - Stage 3: $64 \to 32$ channels, upsample to $296 \times 296$
  - Stage 4: $32 \to 16$ channels, upsample to $518 \times 518$
- **Auxiliary Heads**: Intermediate deep supervision heads at $148 \times 148$ and $296 \times 296$ preventing gradient decay.

### 2.5 Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_D \mathcal{L}_{\text{dice}} + \lambda_B \mathcal{L}_{\text{boundary}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_I \mathcal{L}_{\text{image}}$$
- **Differentiable Sobel Boundary Loss**:
  $$\mathcal{L}_{\text{boundary}} = \frac{1}{|\Omega|} \sum_{u,v} \left| \|\nabla M_{\text{pred}}(u,v)\|_2 - \|\nabla M_{\text{gt}}(u,v)\|_2 \right|$$

### 2.6 Deterministic 9-State Fail-Safe Engine
Predictions are evaluated through a 9-state fail-safe state machine:
- `NORMAL` $\rightarrow$ `ACCEPT`
- `ANOMALY_DETECTED` $\rightarrow$ `REJECT`
- `UNCERTAIN` $\rightarrow$ `HUMAN_REVIEW`
- `OOD` $\rightarrow$ `QUARANTINE`
- `IMAGE_QUALITY_FAILURE` $\rightarrow$ `HOLD`
- `MODEL_UNAVAILABLE` $\rightarrow$ `HOLD`
- `CALIBRATION_INVALID` $\rightarrow$ `QUARANTINE`
- `COMMUNICATION_FAILURE` $\rightarrow$ `HOLD`
- `HUMAN_REVIEW_REQUIRED` $\rightarrow$ `HUMAN_REVIEW`

---

## 3. Data Flow Architecture

```
Camera Frame (518x518x3)
         |
         v
[Ingress Validation: Dim, Verify, Lux, MTF Focus]
         |
         +---------------------------------------+
         |                                       |
         v                                       v
[DINOv2 ViT-S/14 Foundation]            [CLIP ViT-B/16 Encoder]
(Blocks 4, 8, 12 Extract)               (Candidate Prompt Cosine)
         |                                       |
         +-------------------+                   |
         |                   |                   |
         v                   v                   |
[Coreset Memory Bank]    [FiLM SE Decoder]       |
  (M_mem 37x37)           (M_neural 518x518)     |
         |                   |                   |
         +---------+---------+                   |
                   v                             |
        [Residual Spatial Fusion]                |
           (M_final 518x518)                     |
                   |                             |
                   v                             |
     [Conformal & OOD Reliability Engine] <------+
                   |
                   v
     [9-State Fail-Safe Output Schema]
```
