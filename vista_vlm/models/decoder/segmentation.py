import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention for small-defect sensitivity."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ConvBlock(nn.Module):
    """Residual Convolutional Block with Squeeze-and-Excitation Attention."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU()
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU()
        )
        self.se = ChannelAttention(out_channels) if use_se else nn.Identity()
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + res


class ProgressiveSegmentationDecoder(nn.Module):
    """
    Progressive Convolutional Segmentation Decoder with Multi-Scale Skip Connections (v4.0 Architecture).
    Upsamples spatial feature maps from grid resolution (e.g. 37x37)
    progressively back to the full image resolution (e.g. 518x518).
    Supports:
    1. Lateral multi-scale skip projections from intermediate backbone blocks (4, 8)
       to restore high-frequency boundary details and hairline cracks.
    2. Deep supervision auxiliary heads at intermediate resolutions (148x148 and 296x296)
       to prevent vanishing gradients through progressive convolutional stages.
    3. Optional spatial guide map projection (e.g. from nominal reference memory)
       to ground learned boundaries in empirical feature distance evidence.
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dims: List[int] = [128, 64, 32, 16],
        target_size: int = 518,
        dropout: float = 0.1,
        skip_channels: int = 384,
        guide_in_channels: int = 1,
    ):
        super().__init__()
        self.target_size = (target_size, target_size)

        layers = []
        curr_in = in_channels
        for h_dim in hidden_dims:
            layers.append(ConvBlock(curr_in, h_dim, dropout=dropout, use_se=True))
            curr_in = h_dim
        self.stages = nn.ModuleList(layers)

        # Multi-scale lateral skip projections for high-fidelity spatial detail
        self.skip_proj_8 = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_dims[0], kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dims[0]), num_channels=hidden_dims[0]),
            nn.GELU()
        )
        self.skip_proj_4 = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_dims[1], kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dims[1]), num_channels=hidden_dims[1]),
            nn.GELU()
        )

        # Optional spatial guide map projection (e.g. memory distance guide)
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

        # Auxiliary heads for deep supervision
        self.aux_head_148 = nn.Conv2d(hidden_dims[1], 1, kernel_size=1)
        self.aux_head_296 = nn.Conv2d(hidden_dims[2], 1, kernel_size=1)

        self.final_head = nn.Conv2d(curr_in, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        guide_map: Optional[torch.Tensor] = None,
        out_size: Optional[Tuple[int, int]] = None,
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Fused features [B, in_channels, H_p, W_p]
            skips: Optional list of intermediate backbone features [skip_block4, skip_block8]
            guide_map: Optional empirical memory distance guide [B, 1, H_p, W_p] or [B, 1, H, W]
            out_size: Optional target size override (H, W)
            return_aux: Whether to return intermediate auxiliary logits for deep supervision
        Returns:
            pixel_logits: [B, 1, H, W] (or tuple (pixel_logits, aux_logits_list) if return_aux=True)
        """
        target = out_size or self.target_size
        aux_logits = []

        # Inject empirical guide map if provided
        if guide_map is not None:
            if guide_map.shape[-2:] != x.shape[-2:]:
                guide_down = F.interpolate(guide_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:
                guide_down = guide_map
            x = x + self.guide_proj(guide_down)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                # Progressive 2x upsampling
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
                # Inject lateral skip connections if provided
                if skips is not None and len(skips) >= 2:
                    if i == 0:
                        # Fuse Block 8 features at 74x74
                        s8 = self.skip_proj_8(skips[1])
                        s8_up = F.interpolate(s8, size=x.shape[-2:], mode="bilinear", align_corners=False)
                        x = x + s8_up
                    elif i == 1:
                        # Fuse Block 4 features at 148x148 (rich low-level edges)
                        s4 = self.skip_proj_4(skips[0])
                        s4_up = F.interpolate(s4, size=x.shape[-2:], mode="bilinear", align_corners=False)
                        x = x + s4_up

                if return_aux:
                    if i == 1:
                        aux_logits.append(self.aux_head_148(x))
                    elif i == 2:
                        aux_logits.append(self.aux_head_296(x))
            else:
                # Final upsampling to exact target size
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)

        logits = self.final_head(x)

        if return_aux:
            return logits, aux_logits
        return logits
