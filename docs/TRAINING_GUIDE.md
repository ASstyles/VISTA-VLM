# VISTA-VLM V5.0 Training, Memory Fitting & Calibration Guide

**Document ID:** `V5-DOC-TRN-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Overview of V5.0 Training Paradigm

VISTA-VLM V5.0 combines two complementary learning paradigms:
1. **Parametric Self-Supervised Neural Training**: The pre-trained DINOv2 ViT-S/14 backbone is frozen. The FiLM fusion module, Progressive Squeeze-and-Excitation (SE) segmentation decoder, lateral skip projections, deep auxiliary heads, and classification head are trained jointly using procedural synthetic anomalies.
2. **Non-Parametric Nominal Memory Fitting**: Extracts multi-scale patch features (Blocks 4 & 8) from normal training images and builds a compact coreset memory bank ($M = 1,500$ prototypes). This grounds the neural decoder and eliminates spatial localization collapse.

---

## 2. Unified Loss Formulation & Hyperparameters

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_D \mathcal{L}_{\text{dice}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_B \mathcal{L}_{\text{boundary}}(M_{\text{pred}}, M_{\text{gt}}) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_I \mathcal{L}_{\text{bce}}(S(x), y)$$

| Hyperparameter | Default Value | Description |
| :--- | :---: | :--- |
| `learning_rate` | `1e-4` | AdamW optimizer learning rate |
| `weight_decay` | `1e-4` | L2 weight regularization |
| `batch_size` | `4` | Training batch size |
| `image_size` | `518` | Input resolution ($37 \times 37$ tokens for patch size 14) |
| `lambda_dice` | `1.0` | Weight for Smooth Soft Dice segmentation loss |
| `lambda_boundary` | `0.5` | Weight for differentiable Sobel Edge Boundary Loss |
| `lambda_aux` | `0.3` | Weight for intermediate deep supervision heads |
| `lambda_image` | `0.5` | Weight for global image-level Binary Cross-Entropy loss |
| `synthetic_prob` | `0.5` | Probability of injecting synthetic anomaly per training sample |
| `coreset_size` | `1500` | Target number of nominal prototype vectors in memory bank |
| `w_memory` | `0.5` | Memory-guided spatial residual weighting parameter |

---

## 3. End-to-End Training and Calibration Command

```bash
python scripts/train_and_calibrate.py \
  --category bottle \
  --epochs 5 \
  --batch_size 4 \
  --lr 1e-4 \
  --alpha 0.05
```

### Script Execution Stages:
1. **Nominal Image Loading**: Loads flawless reference images from the training directory.
2. **Neural Decoder Fine-Tuning**: Trains the progressive SE decoder using procedural CutPaste and Perlin anomaly masks with Sobel boundary loss.
3. **Coreset Memory Fitting**: Calls `model.fit_memory_bank(train_loader, max_patches=1500)` to populate the reference patch matrix.
4. **Checkpoint Serialization**: Persists model weights, architecture hyperparameters, and the `memory_bank` tensor to `checkpoints/vista_mvtec_{category}.pth`.
5. **Split Conformal Calibration**: Computes non-conformity scores and conformal threshold $\hat{q}$ on held-out normal images, outputting `checkpoints/calibration_mvtec.json`.
6. **Mahalanobis OOD Fitting**: Fits nominal feature mean $\boldsymbol{\mu}$ and Ledoit-Wolf covariance matrix $\mathbf{\Sigma}^{-1}$, outputting `checkpoints/ood_mvtec.json`.
