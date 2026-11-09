# VISTA-VLM v4.0 System Architecture Specification

## 1. Overview & Core Tenets

VISTA-VLM (**Vision Intelligence through Spatial and Textual Alignment**) is designed around four architectural tenets:
1. **Separation of Detection from Semantic Classification**: Dense defect localization is driven by intermediate patch features from a self-supervised visual foundation model (DINOv2), while semantic candidate identification and operator explanations are governed by a multimodal vision-language model (CLIP).
2. **Multi-Scale Spatial-Semantic Harmony**: Dense spatial representations from early layers (Block 4) and semantic abstractions from deep layers (Block 8, 12) are fused via global FiLM conditioning and progressive convolutional decoding with Squeeze-and-Excitation (SE) channel attention.
3. **Statistical Reliability & Dual-Risk Calibration**: No anomaly score is emitted without an accompanying split conformal calibration guarantee ($\alpha = 0.05$), a regularized Mahalanobis Out-of-Distribution (OOD) distance test, and separated false-pass / false-alarm risk profiling.
4. **Selective Prediction with Human-in-the-Loop Escalation**: Autonomous decisions are committed strictly when conformal certainty is verified. Ambiguous border cases within the uncertainty band $[\tau_c - \delta, \tau_c + \delta]$ or OOD inputs are routed to visual inspection operators.

---

## 2. Component Specifications

### 2.1 Visual Foundation Encoder
- **Backbone**: DINOv2 ViT-S/14 (`dinov2_vits14`) (ViT-B/14 supported as secondary scaled variant)
- **Parameters**: 22,050,432 (All frozen, `requires_grad = False`)
- **Input Dimension**: $B \times 3 \times 518 \times 518$
- **Patch Size**: $14 \times 14$
- **Spatial Grid**: $37 \times 37 = 1369$ tokens
- **Intermediate Taps**: Transformer blocks 4, 8, and 12
- **Output Tensors**:
  - `patch_features`: 3 spatial feature maps $[B, 384, 37, 37]$
  - `cls_features`: 3 global tokens $[B, 384]$
  - `global_cls`: Final layer CLS vector $[B, 384]$

### 2.2 Global-Local Feature Fusion
Supports 5 modular fusion configurations:
- **FiLM Modulation (Default)**: Modulates concatenated local patch features ($3 \times 384 = 1152$ channels) with affine parameters $[\boldsymbol{\gamma}, \boldsymbol{\beta}]$ computed from global CLS tokens.
  $$\mathbf{F}_{\text{fused}} = \mathbf{F}_{\text{local}} + \text{Conv}_{3\times3}\left((1 + \boldsymbol{\gamma}) \odot \mathbf{F}_{\text{local}} + \boldsymbol{\beta}\right)$$
- **Gated Feature Fusion**: Element-wise sigmoid gating from global context.
- **Cross-Attention Fusion**: Multi-head cross-attention querying spatial features against global tokens.
- **Concat-Projection**: Concatenation of local and spatially broadcast global tokens followed by linear projection.
- **Local-Only**: Independent convolutional spatial feature refinement without CLS conditioning.

### 2.3 Progressive Segmentation Decoder with SE Attention
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
- **Output**: Anomaly probability map $A(x) \in [0, 1]^{B \times 1 \times 518 \times 518}$.

### 2.4 Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_D \mathcal{L}_{\text{dice}} + \lambda_B \mathcal{L}_{\text{boundary}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_I \mathcal{L}_{\text{image}}$$
- **Differentiable Sobel Boundary Loss**:
  $$\mathcal{L}_{\text{boundary}} = \frac{1}{|\Omega|} \sum_{u,v} \left| \|\nabla M_{\text{pred}}(u,v)\|_2 - \|\nabla M_{\text{gt}}(u,v)\|_2 \right|$$

### 2.5 Multimodal Language Verifier
- **Model**: Frozen CLIP ViT-B/16 (Image & Text encoders)
- **Token Cache**: In-memory caching of pre-computed text embeddings across candidate prompts.
- **Candidate Ranking**: Cosine similarity against category-tailored defect prompt ensembles.
- **Grounded Explanations**: Generates structured, deterministic explanations referencing exact spatial coordinates, bounding box metrics, and margin scores without LLM hallucinations.

---

## 3. Reliability & Calibration Layer

### 3.1 Split Conformal Calibration
- Calibrated on independent nominal validation images ($n$ samples).
- Conformal quantile:
  $$q = \min\left(1.0, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$
- Guarantees marginal false-alarm rate $\le \alpha$ ($0.05$) under exchangeability.

### 3.2 Regularized Shrinkage Mahalanobis OOD Detection
- Fits nominal centroid $\boldsymbol{\mu}_k$ and regularized inverse covariance matrix $\mathbf{\Sigma}_{\text{shrunk}}^{-1}$.
- Tests incoming images against Mahalanobis distance threshold $\tau_{\text{ood}}$ (98th percentile).
- Fully persisted in JSON checkpoints, eliminating uncalibrated metric fallbacks.

### 3.3 Dual-Risk Selective Prediction
- Evaluates false-pass (defect escape) and false-alarm (overkill) risks separately.
- Computes Wilson score 95% confidence intervals on observed rates.
- Flags sample size validity: requires $N_{\text{def}} \ge 300$ for certified $\le 1\%$ escape claims.

---

## 4. API & Production Integration

- **FastAPI Core**: Asynchronous non-blocking web server with Pydantic v2 schemas.
- **Health Probes**: Decoupled `/health/live` and `/health/ready` endpoints conforming to Kubernetes specifications.
- **Security & Integrity**: Magic-byte MIME verification rejecting non-image payloads with HTTP 415; 20MB maximum payload caps.
- **Frontend Studio**: React 19 + TypeScript + Vite interactive research lab and production dashboard.
