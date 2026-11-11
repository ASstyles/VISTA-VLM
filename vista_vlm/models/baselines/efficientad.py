import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from vista_vlm.utils.registry import BASELINE_REGISTRY

class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeatureAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        z = self.encoder(x)
        dec = self.decoder(z)
        if dec.shape[-2:] != orig_size:
            dec = F.interpolate(dec, size=orig_size, mode="bilinear", align_corners=False)
        return dec

@BASELINE_REGISTRY.register("efficientad")
class EfficientAD(nn.Module):
    """
    EfficientAD Baseline (Phase 3).
    Composed of:
    - Pretrained frozen Teacher Network
    - Trained Student Network (mimics teacher on normal patches)
    - Autoencoder (reconstructs normal teacher features)
    """
    def __init__(self, image_size: int = 518, feature_dim: int = 64):
        super().__init__()
        self.image_size = image_size
        self.teacher = SmallCNN(in_channels=3, out_channels=feature_dim)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.student = SmallCNN(in_channels=3, out_channels=feature_dim)
        self.autoencoder = FeatureAutoencoder(in_channels=feature_dim)

    def fit_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        self.teacher.eval()
        with torch.no_grad():
            t_feat = self.teacher(x)

        s_feat = self.student(x)
        ae_recon = self.autoencoder(t_feat)

        # Distance between teacher and student
        loss_st = F.mse_loss(s_feat, t_feat)
        # Distance between teacher and autoencoder reconstruction
        loss_ae = F.mse_loss(ae_recon, t_feat)

        loss = loss_st + loss_ae
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            t_feat = self.teacher(x)
            s_feat = self.student(x)
            ae_recon = self.autoencoder(t_feat)

            # Feature differences
            diff_st = torch.mean((t_feat - s_feat) ** 2, dim=1, keepdim=True)
            diff_ae = torch.mean((t_feat - ae_recon) ** 2, dim=1, keepdim=True)

            diff_map = diff_st + diff_ae
            anomaly_map = F.interpolate(diff_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            image_score = anomaly_map.view(x.shape[0], -1).max(dim=1)[0]

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
