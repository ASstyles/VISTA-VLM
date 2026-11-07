# VISTA-VLM System Architecture Specification

## 1. Overview and Core Philosophy

VISTA-VLM is designed around three architectural tenets:
1. **Separation of Detection from Semantic Classification**: Defect localization is driven by dense self-supervised visual patch features (DINOv2), while semantic identification is handled by multimodal vision-language models (CLIP).
2. **Explicit Reliability and Uncertainty Quantification**: No neural anomaly score is emitted without an accompanying split conformal calibration guarantee ($\alpha = 0.05$) and an Out-of-Distribution (OOD) distance check.
3. **Selective Prediction**: Automated decisions are committed only when conformal certainty is verified; uncertain and anomalous border cases are safely escalated for human operator review.

---

## 2. Component Specifications

### 2.1 Visual Foundation Backbone
- **Model**: DINOv2 ViT-S/14
- **Parameters**: 22,050,432 (All frozen, `requires_grad = False`)
- **Input Dimension**: $B \times 3 \times 518 \times 518$
- **Patch Size**: $14 \times 14$
- **Spatial Grid**: $37 \times 37 = 1369$ tokens
- **Intermediate Taps**: Blocks 4, 8, and 12
- **Output Tensors**:
  - `patch_features`: List of 3 tensors, each $[B, 384, 37, 37]$
  - `cls_features`: List of 3 tensors, each $[B, 384]$
  - `global_cls`: Final CLS token $[B, 384]$

### 2.2 Global-Local Feature Fusion (FiLM)
- **Concatenated Local Features**: $3 \times 384 = 1152$ channels
- **Spatial Projection**: $1152 \to 256$ via $1 \times 1$ Conv + GroupNorm(16) + GELU
- **FiLM Modulation**:
  - Linear MLP: $1152 \to 512$
  - Scale factor: $\gamma \in \mathbb{R}^{B \times 256 \times 1 \times 1}$
  - Shift factor: $\beta \in \mathbb{R}^{B \times 256 \times 1 \times 1}$
  - Modulated features: $F_{film} = \gamma \odot F_{proj} + \beta$
- **Residual Refinement**: $F_{out} = F_{proj} + \text{Conv}_{3\times 3}(\text{GELU}(\text{GroupNorm}(F_{film}))) \in \mathbb{R}^{B \times 256 \times 37 \times 37}$

### 2.3 Progressive Convolutional Segmentation Decoder
- **Stage 1**: $256 \to 128$ channels, bilinear upsample $37 \times 37 \to 74 \times 74$
- **Stage 2**: $128 \to 64$ channels, bilinear upsample $74 \times 74 \to 148 \times 148$
- **Stage 3**: $64 \to 32$ channels, bilinear upsample $148 \times 148 \to 296 \times 296$
- **Stage 4**: $32 \to 16$ channels, bilinear upsample $296 \times 296 \to 518 \times 518$
- **Head**: $16 \to 1$ channel conv yielding pixel logits $[B, 1, 518, 518]$
- **Activation**: $\sigma(\cdot)$ yielding anomaly probability map $A(x) \in [0, 1]^{B \times 1 \times 518 \times 518}$

### 2.4 Hybrid Image Anomaly Scoring Head
- **Top-k Pixel Pooling**: Average of top 1% highest-scoring pixels in $A(x)$
- **Global CLS Head**: 2-layer MLP ($384 \to 128 \to 1$) with Sigmoid activation
- **Combined Score**: $s(x) = 0.5 \cdot s_{global} + 0.5 \cdot s_{topk} \in [0, 1]$

### 2.5 Vision-Language Multimodal Verifier
- **Model**: Frozen CLIP ViT-B/16 (Image Transformer + Text Transformer)
- **Embedding Dimension**: 512 (normalized unit sphere)
- **ROI Crop Size**: $224 \times 224 \times 3$
- **Similarity Metric**: Cosine similarity $\rho = e_{crop}^\top e_{prompt}$
- **Ambiguity Margin**: $\Delta = \rho_{top1} - \rho_{top2} < 0.08$ flags descriptive ambiguity.

---

## 3. Reliability & Calibration Layer

### 3.1 Split Conformal Calibration
- Held-out nominal calibration scores $\{s_1, \dots, s_n\}$
- Level $q = \min(1.0, \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n})$ for target false-alarm rate $\alpha = 0.05$
- Threshold $\tau_c = \text{Quantile}(s_{cal}, q)$
- Empirical p-value: $p(x) = \frac{1 + \sum_{i=1}^n \mathbb{I}(s_i \ge s(x))}{n + 1}$

### 3.2 Feature Prototype Out-of-Distribution Detector
- Nominal Prototype: $\mu = \frac{1}{n} \sum_{i=1}^n z_{cls}^{(i)}$
- Test distance: $d = 1.0 - \frac{z_{cls}(x)^\top \mu}{\|z_{cls}(x)\|_2 \|\mu\|_2}$
- Calibrated OOD threshold: 98th percentile of nominal calibration distances.

### 3.3 Selective Prediction Uncertainty Interval
- Uncertainty band: $[\tau_c - 0.04, \tau_c + 0.04]$
- Any score within this band triggers automated escalation to a human review queue.

---

## 4. API & Systems Integration

- **FastAPI Core**: Asynchronous non-blocking web server with Pydantic v2 validation.
- **Base64 Serialization**: Dense anomaly heatmaps (JET colormap) and binary segmentation masks are serialized to base64 strings for single-payload web delivery.
- **Decoupled Operator Feedback**: Human review decisions submitted via `/feedback` are appended to an immutable audit directory (`benchmarks/feedback/*.json`), preventing unverified runtime weight corruption.
