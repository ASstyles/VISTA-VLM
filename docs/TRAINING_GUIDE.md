# VISTA-VLM v4.0 Training & Fine-Tuning Guide

## 1. Overview

VISTA-VLM v4.0 is trained in a self-supervised nominal regime using procedural synthetic anomalies generated on-the-fly from normal manufacturing reference images. The pre-trained DINOv2 visual backbone is frozen, while the Global-Local FiLM fusion module, Progressive Squeeze-and-Excitation Segmentation Decoder, lateral skip projections, deep supervision auxiliary heads, and global classification head are trained jointly.

---

## 2. Loss Functions and Hyperparameters

VISTA-VLM optimizes a composite multi-task objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_D \mathcal{L}_{\text{dice}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_B \mathcal{L}_{\text{boundary}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_I \mathcal{L}_{\text{bce}}(S(x), y)$$

| Hyperparameter | Default Value | Description |
|:---|---:|:---|
| `learning_rate` | `1e-4` | AdamW optimizer initial learning rate |
| `weight_decay` | `1e-4` | L2 weight regularization |
| `batch_size` | `4` | Batch size (gradient accumulation supported) |
| `image_size` | `518` | Input resolution ($37 \times 37$ tokens for patch size 14) |
| `lambda_dice` | `1.0` | Weight for Smooth Soft Dice segmentation loss |
| `lambda_boundary` | `0.5` | Weight for differentiable Sobel Edge Boundary Loss |
| `lambda_aux` | `0.3` | Weight for intermediate deep supervision heads |
| `lambda_image` | `0.5` | Weight for global image-level Binary Cross-Entropy loss |
| `synthetic_prob` | `0.5` | Probability of injecting synthetic anomaly per training sample |

---

## 3. Training Commands

### 3.1 Standard Training & Calibration Run
```bash
python scripts/train_and_calibrate.py --category bottle --epochs 5 --batch_size 4 --lr 1e-4
```

### 3.2 Key Training Script Stages:
1. **Nominal Data Loader**: Loads flawless training images from `data/processed/mvtec/{category}/train/good`.
2. **On-The-Fly Anomaly Generation**: `SyntheticAnomalyGenerator` blends CutPaste textures and Perlin noise perturbations into normal samples.
3. **Deep Supervision Forward Pass**: Computes auxiliary logits at $148 \times 148$ and $296 \times 296$ intermediate resolutions alongside the primary $518 \times 518$ map.
4. **Checkpoint Serialization**: Saves full model weights, optimizer state, and architecture version metadata to `checkpoints/vista_mvtec_{category}.pth`.
5. **Split Conformal Calibration**: Computes non-conformity quantile $\tau_c$ on held-out normal validation set (`val_calibration/good`) and saves to `checkpoints/calibration_mvtec.json`.
6. **OOD Covariance Fitting**: Fits Ledoit-Wolf shrinkage covariance matrix $\mathbf{\Sigma}^{-1}$ and prototype $\boldsymbol{\mu}$, saving full parameters to `checkpoints/ood_mvtec.json`.
7. **Selective Risk Verification**: Computes uncertainty band $\delta$, empirical false-pass rate, and Wilson 95% confidence intervals on benchmark test split.
