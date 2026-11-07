import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from vista_vlm.models.backbones.dinov2 import DINOv2Backbone, dinov2_vits14, dinov2_vitb14
from vista_vlm.models.fusion.global_local import GlobalLocalFusion
from vista_vlm.models.decoder.segmentation import ProgressiveSegmentationDecoder
from vista_vlm.config.schema import ModelConfig
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.model")

class BinaryFocalLoss(nn.Module):
    """Focal Loss for addressing extreme foreground-background class imbalance in defect segmentation."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class DiceLoss(nn.Module):
    """Smooth Dice Loss for binary segmentation masks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: sigmoid probabilities [B, 1, H, W]
        # target: binary ground truth [B, 1, H, W]
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class VISTAAnomalyModel(nn.Module):
    """
    VISTA Anomaly Localization Model (Phase 8).
    Integrates:
      1. Frozen DINOv2 ViT-S/14 (or ViT-B/14) Foundation Backbone
      2. Multi-layer intermediate patch & CLS extraction (blocks 4, 8, 12)
      3. Global-Local FiLM Feature Fusion
      4. Progressive Convolutional Segmentation Decoder
      5. Hybrid Pixel-Level & Global Image-Level Anomaly Scoring
    """
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        embed_dim: int = 384,
        out_indices: List[int] = [4, 8, 12],
        proj_dim: int = 256,
        image_size: int = 518,
        fusion_type: str = "global_local",
        use_film: bool = True,
        top_k_ratio: float = 0.01,
        global_head_weight: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.top_k_ratio = top_k_ratio
        self.global_head_weight = global_head_weight

        # 1. Visual Backbone
        if backbone_name == "dinov2_vits14":
            self.encoder = dinov2_vits14(img_size=image_size, out_indices=out_indices)
        elif backbone_name == "dinov2_vitb14":
            self.encoder = dinov2_vitb14(img_size=image_size, out_indices=out_indices)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # 2. Global-Local Fusion
        num_layers = len(out_indices)
        in_channels_list = [embed_dim] * num_layers
        self.fusion = GlobalLocalFusion(
            in_channels_list=in_channels_list,
            proj_dim=proj_dim,
            fusion_type=fusion_type,
            use_film_conditioning=use_film
        )

        # 3. Progressive Segmentation Decoder
        self.decoder = ProgressiveSegmentationDecoder(
            in_channels=proj_dim,
            hidden_dims=[128, 64, 32, 16],
            target_size=image_size
        )

        # 4. Global Image-Level Anomaly Head
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.img_bce_loss = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            pixel_logits: [B, 1, H, W]
            anomaly_map: [B, 1, H, W] in [0, 1]
            image_score: [B] in [0, 1]
            global_features: [B, embed_dim]
        """
        B, C, H, W = x.shape

        # 1. Backbone features
        with torch.no_grad():
            feat_dict = self.encoder(x)
            patch_feats = feat_dict["patch_features"] # List of [B, embed_dim, 37, 37]
            cls_feats = feat_dict["cls_features"]     # List of [B, embed_dim]
            global_cls = feat_dict["global_cls"]      # [B, embed_dim]

        # 2. Global-Local Fusion
        fused = self.fusion(patch_feats, cls_feats) # [B, proj_dim, 37, 37]

        # 3. Progressive Segmentation Decoder with multi-scale skip connections
        pixel_logits = self.decoder(fused, skips=patch_feats, out_size=(H, W)) # [B, 1, H, W]
        anomaly_map = torch.sigmoid(pixel_logits)

        # 4. Image-Level Anomaly Score
        # Top-k pixel pooling
        flat_map = anomaly_map.view(B, -1)
        k = max(1, int(flat_map.shape[1] * self.top_k_ratio))
        top_k_vals, _ = torch.topk(flat_map, k, dim=1)
        topk_score = top_k_vals.mean(dim=1) # [B]

        # Global head score
        head_score = self.global_head(global_cls).squeeze(-1) # [B]

        # Hybrid score
        image_score = self.global_head_weight * head_score + (1.0 - self.global_head_weight) * topk_score
        image_score = torch.clamp(image_score, 0.0, 1.0)

        return {
            "pixel_logits": pixel_logits,
            "anomaly_map": anomaly_map,
            "image_score": image_score,
            "global_features": global_cls
        }

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
        lambda_dice: float = 1.0,
        lambda_image: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        L_total = L_seg + lambda_I * L_img
        L_seg = L_focal + lambda_D * Dice
        """
        pixel_logits = pred["pixel_logits"]
        anomaly_map = pred["anomaly_map"]
        image_score = pred["image_score"]

        # Pixel segmentation loss: Focal Loss + Boundary-Aware Dice Loss
        loss_focal = self.focal_loss(pixel_logits, target_mask)
        loss_dice = self.dice_loss(anomaly_map, target_mask)
        loss_seg = loss_focal + lambda_dice * loss_dice

        # Image-level anomaly classification loss
        loss_img = self.img_bce_loss(image_score, target_label)

        loss_total = loss_seg + lambda_image * loss_img

        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_bce": loss_focal,
            "loss_dice": loss_dice,
            "loss_image": loss_img
        }

    def save_checkpoint(self, path: str, extra_metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "image_size": self.image_size,
                "top_k_ratio": self.top_k_ratio,
                "global_head_weight": self.global_head_weight,
            },
            "metadata": extra_metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved successfully to {path}")

    def load_checkpoint(self, path: str, device: str = "cpu", strict: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Checkpoint loaded successfully from {path}")
        return checkpoint.get("metadata", {})
