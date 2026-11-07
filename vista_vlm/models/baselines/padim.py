import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from vista_vlm.models.backbones.dinov2 import dinov2_vits14
from vista_vlm.utils.registry import BASELINE_REGISTRY

@BASELINE_REGISTRY.register("padim")
class PaDiM(nn.Module):
    """
    PaDiM: Patch Distribution Modeling (Phase 3).
    Models spatial patch positions using multivariate Gaussian distributions.
    """
    def __init__(self, embed_dim: int = 128, image_size: int = 518):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.encoder = dinov2_vits14(img_size=image_size, out_indices=[4])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.mean: Optional[torch.Tensor] = None
        self.inv_cov: Optional[torch.Tensor] = None

    def fit(self, dataloader: torch.utils.data.DataLoader, device: str = "cpu"):
        self.encoder.to(device)
        feature_list = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["image"].to(device)
                feat = self.encoder(x)["patch_features"][0] # [B, C, 37, 37]
                if feat.shape[1] > self.embed_dim:
                    feat = feat[:, :self.embed_dim, :, :]
                feature_list.append(feat.cpu())

        all_feats = torch.cat(feature_list, dim=0) # [N, C, H_p, W_p]

        # Compute empirical mean and variance per spatial patch position
        self.mean = torch.mean(all_feats, dim=0) # [C, H_p, W_p]
        variance = torch.var(all_feats, dim=0, unbiased=True) + 0.01 # [C, H_p, W_p]
        self.inv_cov = 1.0 / torch.sqrt(variance) # Diagonal standard deviation scale

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM parameters not fitted. Call fit() first.")

        device = x.device
        self.encoder.to(device)
        mean = self.mean.to(device).unsqueeze(0) # [1, C, H_p, W_p]
        inv_scale = self.inv_cov.to(device).unsqueeze(0) # [1, C, H_p, W_p]

        with torch.no_grad():
            feat = self.encoder(x)["patch_features"][0]
            if feat.shape[1] > self.embed_dim:
                feat = feat[:, :self.embed_dim, :, :]

            # Vectorized normalized Mahalanobis distance
            norm_diff = (feat - mean) * inv_scale
            dist_map = torch.norm(norm_diff, p=2.0, dim=1, keepdim=True) # [B, 1, H_p, W_p]

            anomaly_map = F.interpolate(dist_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            anomaly_map = torch.sigmoid((anomaly_map - anomaly_map.mean()) / (anomaly_map.std() + 1e-6))
            image_score = anomaly_map.view(x.shape[0], -1).max(dim=1)[0]

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
