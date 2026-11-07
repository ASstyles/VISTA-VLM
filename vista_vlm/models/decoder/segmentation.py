import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ProgressiveSegmentationDecoder(nn.Module):
    """
    Progressive Convolutional Segmentation Decoder (Phase 8).
    Upsamples spatial feature maps from grid resolution (e.g. 37x37)
    progressively back to the full image resolution (e.g. 518x518).
    Produces pixel-level anomaly logits and calibrated anomaly heatmaps.
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dims: List[int] = [128, 64, 32, 16],
        target_size: int = 518,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_size = (target_size, target_size)

        layers = []
        curr_in = in_channels
        for h_dim in hidden_dims:
            layers.append(ConvBlock(curr_in, h_dim, dropout=dropout))
            curr_in = h_dim
        self.stages = nn.ModuleList(layers)

        self.final_head = nn.Conv2d(curr_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: Fused features [B, in_channels, H_p, W_p]
            out_size: Optional target size override (H, W)
        Returns:
            pixel_logits: [B, 1, H, W]
        """
        target = out_size or self.target_size

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                # Progressive 2x upsampling
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            else:
                # Final upsampling to exact target size
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)

        logits = self.final_head(x)
        return logits
