"""
VISTA-VLM Core Anomaly Detection Model (V7.0 Research Upgrade).
==============================================================
Unified Architecture combining:
  1. Frozen DINOv2 Foundation Backbone (Intermediate Blocks 4, 8, 12)
  2. Multi-Scale Global-Local FiLM Feature Conditioning
  3. Progressive Convolutional Decoder with Squeeze-and-Excitation Attention & Lateral Skips
  4. Nominal Reference Memory Bank with Exact/Approximate Greedy k-Center Coreset Sampling
  5. Calibrated Multi-Neighbor Spatial Guidance (1-NN, Mean k-NN, Distance-Weighted)
  6. Harmonized Image-Level Scoring Branch (training loss and inference share identical formulations)
  7. Multi-Task Supervision: Focal Loss + Soft Dice + Tversky Loss + Sobel Boundary Loss
  8. Strict Operational Lifecycle State Machine (UNINITIALIZED, LOADING, READY, INVALID, UNAVAILABLE)
"""

from __future__ import annotations

import os
import time
import hashlib
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from vista_vlm.models.backbones.dinov2 import dinov2_vits14, dinov2_vitb14
from vista_vlm.models.fusion.global_local import GlobalLocalFusion
from vista_vlm.models.decoder.segmentation import ProgressiveSegmentationDecoder
from vista_vlm.models.memory_bank import NominalMemoryBank, CoresetAlgorithm
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.model")


class ModelState(str, Enum):
    """Explicit Operational Model State Machine."""
    UNINITIALIZED = "UNINITIALIZED"
    LOADING = "LOADING"
    READY = "READY"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"


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
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss with asymmetric false-negative penalization.
    Forces high recall on fine defect cracks and boundary anomalies by setting alpha > beta.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_flat = pred.contiguous().view(-1)
        t_flat = target.contiguous().view(-1)

        tp = (p_flat * t_flat).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky


class SobelBoundaryLoss(nn.Module):
    """
    Differentiable Sobel Edge Gradient Loss.
    Penalizes diffuse or blurry spatial heatmaps and forces the network
    to predict crisp, sharp defect contours matching ground-truth boundaries.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred_map: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        px = F.conv2d(pred_map, self.sobel_x, padding=1)
        py = F.conv2d(pred_map, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(px ** 2 + py ** 2 + 1e-6)

        tx = F.conv2d(target_mask, self.sobel_x, padding=1)
        ty = F.conv2d(target_mask, self.sobel_y, padding=1)
        target_grad = torch.sqrt(tx ** 2 + ty ** 2 + 1e-6)

        return F.l1_loss(pred_grad, target_grad)


class VISTAAnomalyModel(nn.Module):
    """
    VISTA Anomaly Detection & Dense Localization Model (V7.0 Research Upgrade).
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
        top_k_ratio: float = 0.005,
        global_head_weight: float = 0.20,
        scoring_mode: str = "hybrid",
        freeze_backbone: bool = True,
        memory_fusion_weight: float = 0.65,
        coreset_algorithm: str = "approx_greedy_k_center",
        coreset_max_patches: int = 1500,
        category: str = "generic",
        k_neighbors: int = 3,
        neighbor_aggregation: str = "mean"
    ):
        super().__init__()
        self.image_size = image_size
        self.top_k_ratio = top_k_ratio
        self.global_head_weight = global_head_weight
        self.scoring_mode = scoring_mode
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.backbone_name = backbone_name
        self.memory_fusion_weight = memory_fusion_weight
        self.category = category
        self.state = ModelState.UNINITIALIZED

        # 1. Visual Foundation Backbone
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

        # 2. Global-Local Multi-Scale Fusion
        num_layers = len(out_indices)
        in_channels_list = [embed_dim] * num_layers
        self.fusion = GlobalLocalFusion(
            in_channels_list=in_channels_list,
            proj_dim=proj_dim,
            fusion_type=fusion_type,
            use_film_conditioning=use_film
        )

        # 3. Progressive Segmentation Decoder with SE Attention & Lateral Skips
        self.decoder = ProgressiveSegmentationDecoder(
            in_channels=proj_dim,
            hidden_dims=[128, 64, 32, 16],
            target_size=image_size,
            guide_in_channels=1,
            skip_channels=embed_dim
        )

        # 4. Global Image-Level Anomaly Head
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 5. Nominal Reference Memory Bank
        self.nominal_memory = NominalMemoryBank(
            algorithm=coreset_algorithm,
            max_patches=coreset_max_patches,
            seed=42,
            k_neighbors=k_neighbors,
            neighbor_aggregation=neighbor_aggregation
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.sobel_loss = SobelBoundaryLoss()
        self.img_bce_loss = nn.BCELoss()

        self.state = ModelState.READY

    @property
    def memory_bank(self) -> Optional[torch.Tensor]:
        """Backwards-compatibility property returning memory bank tensor."""
        return self.nominal_memory.patches

    @memory_bank.setter
    def memory_bank(self, value: Optional[torch.Tensor]):
        """Backwards-compatibility property setter for memory bank tensor."""
        self.nominal_memory.patches = value

    @property
    def is_ready(self) -> bool:
        return self.state == ModelState.READY

    def fit_memory_bank(
        self,
        normal_loader,
        device: str = "cpu",
        max_patches: int = 1500,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fits category nominal memory bank from normal reference images.
        Extracts multi-scale representations combining Block 4 (fine structural details)
        and Block 8 (mid-level contextual semantics).
        Computes robust reference normal distribution statistics for calibrated cross-image scaling.
        """
        self.encoder.eval()
        target_cat = category or self.category
        patch_list = []
        with torch.no_grad():
            for batch in normal_loader:
                imgs = batch["image"].to(device)
                feat_dict = self.encoder(imgs)
                p4 = feat_dict["patch_features"][0]
                p8 = feat_dict["patch_features"][1]
                p_cat = torch.cat([p4, p8], dim=1)  # [B, 2*embed_dim, gh, gw]
                B, C, gh, gw = p_cat.shape
                tokens = p_cat.permute(0, 2, 3, 1).reshape(-1, C)
                patch_list.append(tokens.cpu())

        if not patch_list:
            raise ValueError("No patches collected from normal_loader to fit memory bank.")

        all_patches = torch.cat(patch_list, dim=0)
        self.nominal_memory.max_patches = max_patches
        info = self.nominal_memory.fit(all_patches, category=target_cat, device=device)
        self.category = target_cat
        self.state = ModelState.READY
        return info

    def compute_memory_guide(
        self,
        patch_feats: List[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes nearest-neighbor patch distance map from nominal reference memory.
        Uses FIXED training-calibrated normal reference stats rather than per-image min-max distortion.
        Returns:
            guide_map: Reference-calibrated [B, 1, gh, gw] in [0, 1] for decoder conditioning
            raw_dist_map: Continuous un-normalized Euclidean distance [B, 1, gh, gw]
            raw_scores: Un-normalized max distance per image [B]
        """
        if not self.nominal_memory.is_fitted or len(patch_feats) < 2:
            return None, None, None

        p4 = patch_feats[0]
        p8 = patch_feats[1]
        p_cat = torch.cat([p4, p8], dim=1)  # [B, 2*embed_dim, gh, gw]
        B, C, gh, gw = p_cat.shape
        patches = p_cat.permute(0, 2, 3, 1).reshape(B, gh * gw, C)

        raw_dist_map, calibrated_map, raw_scores = self.nominal_memory.compute_distance_map(
            patches, spatial_shape=(gh, gw)
        )
        return calibrated_map, raw_dist_map, raw_scores

    def forward(
        self,
        x: torch.Tensor,
        guide_map: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass with unified scoring branch.
        """
        B, C, H, W = x.shape

        # 1. Backbone features
        with torch.no_grad():
            feat_dict = self.encoder(x)
            patch_feats = feat_dict["patch_features"]  # List of [B, embed_dim, 37, 37]
            cls_feats = feat_dict["cls_features"]      # List of [B, embed_dim]
            global_cls = feat_dict["global_cls"]       # [B, embed_dim]

        # Compute empirical memory guide if memory bank is fitted and guide not provided
        raw_mem_scores = None
        raw_dist_map = None
        if guide_map is None and self.nominal_memory.is_fitted:
            guide_map, raw_dist_map, raw_mem_scores = self.compute_memory_guide(patch_feats)

        # 2. Global-Local Fusion
        fused = self.fusion(patch_feats, cls_feats)  # [B, proj_dim, 37, 37]

        # 3. Progressive Segmentation Decoder with SE Attention and Skips
        dec_out = self.decoder(
            fused,
            skips=patch_feats,
            guide_map=guide_map,
            out_size=(H, W),
            return_aux=return_aux
        )
        if return_aux:
            pixel_logits, aux_logits = dec_out
        else:
            pixel_logits = dec_out
            aux_logits = None

        neural_map = torch.sigmoid(pixel_logits)

        # 4. Memory-Guided Spatial Residual Fusion
        if guide_map is not None:
            guide_up = F.interpolate(guide_map, size=(H, W), mode="bilinear", align_corners=False)
            anomaly_map = (1.0 - self.memory_fusion_weight) * neural_map + self.memory_fusion_weight * guide_up
        else:
            anomaly_map = neural_map

        # 5. Image-Level Anomaly Score: Harmonized Unified Formulation
        flat_map = anomaly_map.view(B, -1)
        k = max(1, int(flat_map.shape[1] * self.top_k_ratio))
        top_k_vals, _ = torch.topk(flat_map, k, dim=1)
        topk_spatial_score = top_k_vals.mean(dim=1)

        # Head score from DINOv2 global feature representation
        head_score = self.global_head(global_cls).squeeze(-1)

        # If memory bank is active, compute scaled memory score using reference distribution
        if raw_mem_scores is not None and self.nominal_memory.is_fitted:
            ref_min = self.nominal_memory.ref_stats["min"]
            ref_scale = self.nominal_memory.ref_stats["scale_range"]
            cal_mem_score = torch.clamp((raw_mem_scores - ref_min) / ref_scale, 0.0, 1.0)
        else:
            cal_mem_score = topk_spatial_score

        if self.scoring_mode == "memory_max" and raw_mem_scores is not None:
            image_score = cal_mem_score
        elif self.scoring_mode == "memory_topk":
            image_score = topk_spatial_score
        elif self.scoring_mode == "top_k":
            image_score = topk_spatial_score
        elif self.scoring_mode == "max":
            image_score = flat_map.max(dim=1)[0]
        elif self.scoring_mode == "percentile_99":
            idx = int(flat_map.shape[1] * 0.99)
            image_score = torch.sort(flat_map, dim=1)[0][:, idx]
        else:  # "hybrid" / "learned_fusion" (Unified Harmonized Scoring)
            if raw_mem_scores is not None and self.nominal_memory.is_fitted:
                # Fuses memory distance, spatial top-k, and trained global head
                w_head = self.global_head_weight
                w_mem = 0.50 * (1.0 - w_head)
                w_spatial = 0.50 * (1.0 - w_head)
                image_score = w_mem * cal_mem_score + w_spatial * topk_spatial_score + w_head * head_score
            else:
                image_score = self.global_head_weight * head_score + (1.0 - self.global_head_weight) * topk_spatial_score

        image_score = torch.clamp(image_score, 0.0, 1.0)

        out_dict = {
            "pixel_logits": pixel_logits,
            "anomaly_map": anomaly_map,
            "image_score": image_score,
            "raw_scores": raw_mem_scores if raw_mem_scores is not None else image_score,
            "global_features": global_cls
        }
        if aux_logits is not None:
            out_dict["aux_logits"] = aux_logits
        return out_dict

    def compute_loss(
        self,
        pred: Dict[str, Any],
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
        lambda_dice: float = 1.0,
        lambda_tversky: float = 0.5,
        lambda_boundary: float = 0.5,
        lambda_aux: float = 0.3,
        lambda_image: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        pixel_logits = pred["pixel_logits"]
        anomaly_map = pred["anomaly_map"]
        image_score = pred["image_score"]

        # 1. Primary Pixel Segmentation Loss: Focal Loss + Soft Dice + Tversky
        loss_focal = self.focal_loss(pixel_logits, target_mask)
        loss_dice = self.dice_loss(anomaly_map, target_mask)
        loss_tversky = self.tversky_loss(anomaly_map, target_mask)
        loss_seg = loss_focal + lambda_dice * loss_dice + lambda_tversky * loss_tversky

        # 2. Differentiable Sobel Boundary Loss
        loss_boundary = self.sobel_loss(anomaly_map, target_mask)

        # 3. Deep Supervision Auxiliary Loss
        loss_aux = torch.tensor(0.0, device=pixel_logits.device)
        if "aux_logits" in pred and pred["aux_logits"]:
            aux_losses = []
            for aux_log in pred["aux_logits"]:
                aux_target = F.interpolate(target_mask, size=aux_log.shape[-2:], mode="nearest")
                aux_losses.append(self.focal_loss(aux_log, aux_target))
            loss_aux = sum(aux_losses) / len(aux_losses)

        # 4. Image-level anomaly classification loss
        loss_img = self.img_bce_loss(image_score, target_label)

        loss_total = loss_seg + lambda_boundary * loss_boundary + lambda_aux * loss_aux + lambda_image * loss_img

        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_bce": loss_focal,
            "loss_dice": loss_dice,
            "loss_tversky": loss_tversky,
            "loss_boundary": loss_boundary,
            "loss_aux": loss_aux,
            "loss_image": loss_img
        }

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves production-grade checkpoint with full V7.0 metadata, architectural
        specifications, nominal memory bank state, and SHA-256 integrity digest.
        Uses atomic file write to prevent corrupted checkpoint saves on interruption.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = extra_metadata or {}
        meta.update({
            "version": "7.0.0",
            "architecture": "v7.0.0-calibrated-multiscale-memory-residual-fusion",
            "backbone": self.backbone_name,
            "embed_dim": self.embed_dim,
            "out_indices": self.out_indices,
            "category": self.category,
            "coreset_algorithm": self.nominal_memory.algorithm.value,
            "coreset_size": self.nominal_memory.patches.shape[0] if self.nominal_memory.is_fitted else 0,
            "ref_stats": self.nominal_memory.ref_stats,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_state": self.state.value
        })

        checkpoint = {
            "version": "7.0.0",
            "model_state_dict": self.state_dict(),
            "memory_bank_state": self.nominal_memory.state_dict(),
            "memory_bank": self.nominal_memory.patches.cpu() if self.nominal_memory.patches is not None else None,
            "config": {
                "image_size": self.image_size,
                "top_k_ratio": self.top_k_ratio,
                "global_head_weight": self.global_head_weight,
                "scoring_mode": self.scoring_mode,
                "memory_fusion_weight": self.memory_fusion_weight,
                "embed_dim": self.embed_dim,
                "backbone_name": self.backbone_name,
                "k_neighbors": self.nominal_memory.k_neighbors,
                "neighbor_aggregation": self.nominal_memory.neighbor_aggregation
            },
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": meta
        }

        # Atomic write
        tmp_path = f"{path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        logger.info(f"V7 Checkpoint saved to {path} | SHA-256={file_hash[:12]}...")
        return file_hash

    def load_checkpoint(
        self,
        path: str,
        device: str = "cpu",
        strict: bool = False,
        expected_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loads checkpoint with strict validation of architectural compatibility,
        category matching, and memory bank integrity. Fails closed on mismatch.
        """
        if not os.path.exists(path):
            self.state = ModelState.UNAVAILABLE
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        self.state = ModelState.LOADING
        try:
            checkpoint = torch.load(path, map_location=device)
        except Exception as e:
            self.state = ModelState.INVALID
            raise RuntimeError(f"Corrupt checkpoint could not be decoded: {e}")

        # Check category compatibility if expected_category provided
        meta = checkpoint.get("metadata", {})
        saved_cat = meta.get("category", checkpoint.get("config", {}).get("category", None))
        if expected_category and saved_cat and saved_cat != expected_category:
            self.state = ModelState.INVALID
            raise ValueError(
                f"Checkpoint category mismatch: expected '{expected_category}', checkpoint contains '{saved_cat}'"
            )

        # Validate architecture compatibility
        cfg = checkpoint.get("config", {})
        if "embed_dim" in cfg and cfg["embed_dim"] != self.embed_dim:
            self.state = ModelState.INVALID
            raise ValueError(
                f"Architecture mismatch: checkpoint embed_dim={cfg['embed_dim']} vs model embed_dim={self.embed_dim}"
            )

        # Load weights
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load nominal memory bank
        if "memory_bank_state" in checkpoint and checkpoint["memory_bank_state"] is not None:
            self.nominal_memory.load_state_dict(checkpoint["memory_bank_state"], device=device)
        elif "memory_bank" in checkpoint and checkpoint["memory_bank"] is not None:
            self.nominal_memory.patches = checkpoint["memory_bank"].to(device)

        if saved_cat:
            self.category = saved_cat

        self.state = ModelState.READY
        logger.info(f"V7 Checkpoint loaded successfully from {path} | Category: {self.category}")
        return meta
