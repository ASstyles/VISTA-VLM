import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from vista_vlm.utils.registry import BASELINE_REGISTRY

class ReconstructiveSubNetwork(nn.Module):
    """Autoencoder reconstructor."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 32, 4, 2, 1)
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        d1 = F.relu(self.dec1(e3))
        d2 = F.relu(self.dec2(d1))
        d3 = torch.tanh(self.dec3(d2))
        if d3.shape[-2:] != orig_size:
            d3 = F.interpolate(d3, size=orig_size, mode="bilinear", align_corners=False)
        return d3

class DiscriminativeSubNetwork(nn.Module):
    """Discriminative segmenter conditioned on input and reconstruction."""
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, x_recon], dim=1) # [B, 6, H, W]
        h = F.relu(self.conv1(inp))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        logits = self.out(h)
        return logits

@BASELINE_REGISTRY.register("draem")
class DRAEM(nn.Module):
    """
    DRAEM: Discriminatively Trained Reconstruction Anomaly Embedding Model (Phase 3).
    """
    def __init__(self, image_size: int = 518):
        super().__init__()
        self.image_size = image_size
        self.reconstructor = ReconstructiveSubNetwork(in_channels=3)
        self.discriminator = DiscriminativeSubNetwork(in_channels=6)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = self.reconstructor(x)
        logits = self.discriminator(x, recon)
        anomaly_map = torch.sigmoid(logits)
        image_score = anomaly_map.view(x.shape[0], -1).max(dim=1)[0]
        return {
            "reconstruction": recon,
            "pixel_logits": logits,
            "anomaly_map": anomaly_map,
            "image_score": image_score
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(x)

    def compute_loss(
        self,
        x_anom: torch.Tensor,
        x_norm: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        recon = self.reconstructor(x_anom)
        logits = self.discriminator(x_anom, recon)
        anomaly_map = torch.sigmoid(logits)

        # L2 reconstruction loss
        loss_l2 = F.mse_loss(recon, x_norm)
        # Focal/BCE segmentation loss
        loss_bce = F.binary_cross_entropy(anomaly_map, gt_mask)

        loss_total = loss_l2 + loss_bce
        return {
            "loss_total": loss_total,
            "loss_recon": loss_l2,
            "loss_seg": loss_bce
        }
