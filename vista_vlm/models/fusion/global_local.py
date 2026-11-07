import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class GlobalLocalFusion(nn.Module):
    """
    Global-Local Feature Fusion Module (Phase 6).
    Combines multi-layer local spatial features with global context vectors using
    Feature-wise Linear Modulation (FiLM) and residual spatial refinement.
    Supports configurable ablation modes: 'global_local', 'local_only', 'concat_only'.
    """
    def __init__(
        self,
        in_channels_list: List[int], # e.g. [384, 384, 384] for blocks 4, 8, 12
        proj_dim: int = 256,
        fusion_type: str = "global_local", # 'global_local', 'local_only', 'concat_only'
        use_film_conditioning: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_film = use_film_conditioning and (fusion_type == "global_local")
        self.total_in_channels = sum(in_channels_list)
        self.proj_dim = proj_dim

        # 1. Local spatial projection
        self.local_proj = nn.Sequential(
            nn.Conv2d(self.total_in_channels, proj_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=proj_dim),
            nn.GELU()
        )

        # 2. Global context conditioning
        if self.use_film:
            self.global_mlp = nn.Sequential(
                nn.Linear(self.total_in_channels, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim * 2) # gamma and beta
            )
            # Spatial refinement conv
            self.refine_conv = nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        elif self.fusion_type == "concat_only":
            self.concat_proj = nn.Sequential(
                nn.Conv2d(proj_dim + self.total_in_channels, proj_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        else: # 'local_only' ablation
            self.refine_conv = nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

    def forward(
        self,
        patch_features: List[torch.Tensor],
        cls_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            patch_features: List of spatial tensors, each [B, C_i, H_p, W_p]
            cls_features: List of global CLS tensors, each [B, C_i]
        Returns:
            fused_features: [B, proj_dim, H_p, W_p]
        """
        # 1. Concatenate multi-layer local spatial features
        f_local = torch.cat(patch_features, dim=1) # [B, sum(C_i), H_p, W_p]
        f_local_proj = self.local_proj(f_local) # [B, proj_dim, H_p, W_p]

        if self.use_film:
            # 2. Global context modulation
            f_global = torch.cat(cls_features, dim=1) # [B, sum(C_i)]
            film_params = self.global_mlp(f_global) # [B, proj_dim * 2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, proj_dim, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, proj_dim, 1, 1]

            # FiLM modulation: gamma * f + beta
            f_conditioned = (1.0 + gamma) * f_local_proj + beta
            # Residual refinement
            f_fused = f_local_proj + self.refine_conv(f_conditioned)
            return f_fused

        elif self.fusion_type == "concat_only":
            f_global = torch.cat(cls_features, dim=1)
            B, _, H_p, W_p = f_local_proj.shape
            f_global_spatial = f_global.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H_p, W_p)
            f_combined = torch.cat([f_local_proj, f_global_spatial], dim=1)
            return self.concat_proj(f_combined)

        else: # 'local_only'
            return f_local_proj + self.refine_conv(f_local_proj)
