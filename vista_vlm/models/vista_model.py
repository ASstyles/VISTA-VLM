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
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


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
    VISTA Anomaly Localization Model (VISTA-VLM v4.0).
    Integrates:
      1. Frozen DINOv2 ViT-S/14 Foundation Backbone with intermediate extraction
      2. Multi-variant Global-Local Fusion (FiLM, Gated, Cross-Attention, Concat-Proj)
      3. Progressive Segmentation Decoder with Squeeze-and-Excitation, Lateral Skips & Deep Supervision
      4. Hybrid empirical nominal memory residual guidance
      5. Configurable image-level scoring policies (hybrid, top-k, max, 99th percentile)
      6. Multi-task loss: Binary Focal Loss + Soft Dice + Differentiable Sobel Boundary Loss
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
        scoring_mode: str = "hybrid",
        freeze_backbone: bool = True,
        memory_fusion_weight: float = 0.65,
    ):
        super().__init__()
        self.image_size = image_size
        self.top_k_ratio = top_k_ratio
        self.global_head_weight = global_head_weight
        self.scoring_mode = scoring_mode
        self.embed_dim = embed_dim
        self.memory_fusion_weight = memory_fusion_weight

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

        # 3. Progressive Segmentation Decoder with SE Attention & Multi-Scale Skips
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

        # 5. Optional empirical memory bank for hybrid guidance
        self.memory_bank: Optional[torch.Tensor] = None

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.sobel_loss = SobelBoundaryLoss()
        self.img_bce_loss = nn.BCELoss()

    def fit_memory_bank(self, normal_loader, device: str = "cpu", max_patches: int = 1500):
        """
        Fits category nominal memory bank from normal reference images (for Track 1 / Track 4).
        Extracts multi-scale representations combining Block 4 (fine structural edges)
        and Block 8 (mid-level contextual semantics).
        """
        self.encoder.eval()
        patch_list = []
        with torch.no_grad():
            for batch in normal_loader:
                imgs = batch["image"].to(device)
                feat_dict = self.encoder(imgs)
                # Concatenate block 4 (low-level fine edges) and block 8 (mid-level semantics)
                p4 = feat_dict["patch_features"][0]
                p8 = feat_dict["patch_features"][1]
                p_cat = torch.cat([p4, p8], dim=1)  # [B, 2*embed_dim, gh, gw]
                B, C, gh, gw = p_cat.shape
                tokens = p_cat.permute(0, 2, 3, 1).reshape(-1, C)
                patch_list.append(tokens.cpu())

        if patch_list:
            all_patches = torch.cat(patch_list, dim=0)
            total_p = all_patches.shape[0]
            if total_p > max_patches:
                indices = torch.linspace(0, total_p - 1, steps=max_patches).long()
                selected = all_patches[indices]
            else:
                selected = all_patches
            self.memory_bank = selected.to(device)
            logger.info(f"Fitted nominal memory bank with {self.memory_bank.shape[0]} reference patches (Blocks 4 & 8).")

    def compute_memory_guide(self, patch_feats: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes nearest-neighbor patch distance map from nominal reference memory.
        Returns:
            guide_norm: Normalized [B, 1, gh, gw] in [0, 1] for decoder conditioning
            raw_scores: Un-normalized max distance [B] for image-level ranking
        """
        if self.memory_bank is None or len(patch_feats) < 2:
            return None, None

        device = patch_feats[0].device
        mem = self.memory_bank.to(device)
        p4 = patch_feats[0]
        p8 = patch_feats[1]
        p_cat = torch.cat([p4, p8], dim=1)  # [B, 2*embed_dim, gh, gw]
        B, C, gh, gw = p_cat.shape
        patches = p_cat.permute(0, 2, 3, 1).reshape(B, gh * gw, C)

        dist_maps = []
        for b in range(B):
            p_b = patches[b]
            dists = torch.cdist(p_b, mem, p=2.0)
            min_dists, _ = torch.topk(dists, k=3, dim=1, largest=False)
            dist_maps.append(min_dists[:, 0].reshape(1, 1, gh, gw))

        raw_dist = torch.cat(dist_maps, dim=0)  # [B, 1, gh, gw]
        raw_scores = raw_dist.view(B, -1).max(dim=1)[0]  # [B]

        # Normalized guide map per image for decoder injection
        g_min = raw_dist.amin(dim=(-2, -1), keepdim=True)
        g_max = raw_dist.amax(dim=(-2, -1), keepdim=True)
        guide_norm = (raw_dist - g_min) / (g_max - g_min + 1e-6)

        return guide_norm, raw_scores

    def forward(
        self,
        x: torch.Tensor,
        guide_map: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            x: Input images [B, 3, H, W]
            guide_map: Optional empirical memory distance guide [B, 1, H, W] or [B, 1, 37, 37]
            return_aux: Whether to return auxiliary logits for deep supervision
        Returns:
            Dictionary with pixel_logits, anomaly_map, image_score, global_features
        """
        B, C, H, W = x.shape

        # 1. Backbone features
        with torch.no_grad():
            feat_dict = self.encoder(x)
            patch_feats = feat_dict["patch_features"]  # List of [B, embed_dim, 37, 37]
            cls_feats = feat_dict["cls_features"]      # List of [B, embed_dim]
            global_cls = feat_dict["global_cls"]       # [B, embed_dim]

        # Compute empirical memory guide if memory bank fitted and not provided
        raw_mem_scores = None
        if guide_map is None and self.memory_bank is not None:
            guide_map, raw_mem_scores = self.compute_memory_guide(patch_feats)

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

        # 4. Memory-Guided Spatial Residual Fusion (Phase 5 Localization Breakthrough)
        if guide_map is not None:
            guide_up = F.interpolate(guide_map, size=(H, W), mode="bilinear", align_corners=False)
            anomaly_map = (1.0 - self.memory_fusion_weight) * neural_map + self.memory_fusion_weight * guide_up
        else:
            anomaly_map = neural_map

        # 5. Image-Level Anomaly Score
        flat_map = anomaly_map.view(B, -1)
        k = max(1, int(flat_map.shape[1] * self.top_k_ratio))
        top_k_vals, _ = torch.topk(flat_map, k, dim=1)
        topk_score = top_k_vals.mean(dim=1)

        head_score = self.global_head(global_cls).squeeze(-1)

        if self.scoring_mode == "top_k":
            image_score = topk_score
        elif self.scoring_mode == "max":
            image_score = flat_map.max(dim=1)[0]
        elif self.scoring_mode == "percentile_99":
            idx = int(flat_map.shape[1] * 0.99)
            image_score = torch.sort(flat_map, dim=1)[0][:, idx]
        else:  # "hybrid"
            image_score = self.global_head_weight * head_score + (1.0 - self.global_head_weight) * topk_score

        image_score = torch.clamp(image_score, 0.0, 1.0)

        out_dict = {
            "pixel_logits": pixel_logits,
            "anomaly_map": anomaly_map,
            "image_score": image_score,
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
        lambda_boundary: float = 0.5,
        lambda_aux: float = 0.3,
        lambda_image: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        pixel_logits = pred["pixel_logits"]
        anomaly_map = pred["anomaly_map"]
        image_score = pred["image_score"]

        # 1. Primary Pixel Segmentation Loss: Focal Loss + Soft Dice
        loss_focal = self.focal_loss(pixel_logits, target_mask)
        loss_dice = self.dice_loss(anomaly_map, target_mask)
        loss_seg = loss_focal + lambda_dice * loss_dice

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
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "version": "5.0.0",
            "model_state_dict": self.state_dict(),
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "config": {
                "image_size": self.image_size,
                "top_k_ratio": self.top_k_ratio,
                "global_head_weight": self.global_head_weight,
                "scoring_mode": self.scoring_mode,
                "memory_fusion_weight": self.memory_fusion_weight,
            },
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": extra_metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved successfully to {path}")

    def load_checkpoint(self, path: str, device: str = "cpu", strict: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if "memory_bank" in checkpoint and checkpoint["memory_bank"] is not None:
            self.memory_bank = checkpoint["memory_bank"].to(device)
            logger.info(f"Loaded memory bank with {self.memory_bank.shape[0]} reference patches.")
        logger.info(f"Checkpoint loaded successfully from {path}")
        return checkpoint.get("metadata", {})
