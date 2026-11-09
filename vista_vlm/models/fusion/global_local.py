import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class CrossAttentionFusion(nn.Module):
    """Lightweight Cross-Attention: Global CLS Query modulates Local Spatial Tokens."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_local: torch.Tensor, f_global: torch.Tensor) -> torch.Tensor:
        # f_local: [B, C, H, W] -> [B, N, C]
        B, C, H, W = f_local.shape
        N = H * W
        tokens = f_local.flatten(2).transpose(1, 2) # [B, N, C]
        q = self.q_proj(f_global).unsqueeze(1)      # [B, 1, C]
        k = self.k_proj(tokens)                     # [B, N, C]
        v = self.v_proj(tokens)                     # [B, N, C]

        # Reshape for multi-head attention
        q = q.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, 1, head_dim]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N, head_dim]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, num_heads, 1, N]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, C) # [B, 1, C]
        out = self.out_proj(out) # [B, 1, C]

        # Broadcast attention context back across spatial tokens
        modulated = tokens + out
        modulated = self.norm(modulated)
        return modulated.transpose(1, 2).reshape(B, C, H, W)


class GlobalLocalFusion(nn.Module):
    """
    Global-Local Feature Fusion Module (v4.0 Architecture).
    Combines multi-layer local spatial features with global context vectors.
    Supports 5 configurable architectural fusion variants:
      1. 'global_local' / 'film': Feature-wise Linear Modulation
      2. 'concat_proj': Concatenation + 1x1 learned projection
      3. 'gated': Gated Feature Fusion
      4. 'cross_attention': Multi-Head Cross Attention
      5. 'local_only': Pure spatial feature refinement without global vectors
    """
    def __init__(
        self,
        in_channels_list: List[int], # e.g. [384, 384, 384] for blocks 4, 8, 12
        proj_dim: int = 256,
        fusion_type: str = "global_local", # 'global_local', 'film', 'concat_proj', 'gated', 'cross_attention', 'local_only'
        use_film_conditioning: bool = True,
    ):
        super().__init__()
        # Standardize fusion type
        if fusion_type in ("film", "global_local"):
            self.fusion_type = "film" if use_film_conditioning else "local_only"
        elif fusion_type in ("concat_only", "concat_proj"):
            self.fusion_type = "concat_proj"
        else:
            self.fusion_type = fusion_type

        self.total_in_channels = sum(in_channels_list)
        self.proj_dim = proj_dim

        # 1. Local spatial projection
        self.local_proj = nn.Sequential(
            nn.Conv2d(self.total_in_channels, proj_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=proj_dim),
            nn.GELU()
        )

        # 2. Global context conditioning module
        if self.fusion_type == "film":
            self.global_mlp = nn.Sequential(
                nn.Linear(self.total_in_channels, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim * 2) # gamma and beta
            )
            self.refine_conv = nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        elif self.fusion_type == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.total_in_channels, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Sigmoid()
            )
            self.refine_conv = nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        elif self.fusion_type == "cross_attention":
            self.global_proj = nn.Linear(self.total_in_channels, proj_dim)
            self.cross_attn = CrossAttentionFusion(dim=proj_dim, num_heads=4)
            self.refine_conv = nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        elif self.fusion_type == "concat_proj":
            self.concat_proj = nn.Sequential(
                nn.Conv2d(proj_dim + self.total_in_channels, proj_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=16, num_channels=proj_dim),
                nn.GELU()
            )

        else: # 'local_only'
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
        f_local = torch.cat(patch_features, dim=1) # [B, sum(C_i), H_p, W_p]
        f_local_proj = self.local_proj(f_local)   # [B, proj_dim, H_p, W_p]

        if self.fusion_type == "film":
            f_global = torch.cat(cls_features, dim=1) # [B, sum(C_i)]
            film_params = self.global_mlp(f_global)   # [B, proj_dim * 2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, proj_dim, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, proj_dim, 1, 1]

            f_conditioned = (1.0 + gamma) * f_local_proj + beta
            return f_local_proj + self.refine_conv(f_conditioned)

        elif self.fusion_type == "gated":
            f_global = torch.cat(cls_features, dim=1) # [B, sum(C_i)]
            gate = self.gate_mlp(f_global).unsqueeze(-1).unsqueeze(-1) # [B, proj_dim, 1, 1]
            f_gated = gate * f_local_proj
            return f_local_proj + self.refine_conv(f_gated)

        elif self.fusion_type == "cross_attention":
            f_global = torch.cat(cls_features, dim=1)
            f_global_proj = self.global_proj(f_global) # [B, proj_dim]
            f_attn = self.cross_attn(f_local_proj, f_global_proj)
            return f_local_proj + self.refine_conv(f_attn)

        elif self.fusion_type == "concat_proj":
            f_global = torch.cat(cls_features, dim=1)
            B, _, H_p, W_p = f_local_proj.shape
            f_global_spatial = f_global.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H_p, W_p)
            f_combined = torch.cat([f_local_proj, f_global_spatial], dim=1)
            return self.concat_proj(f_combined)

        else: # 'local_only'
            return f_local_proj + self.refine_conv(f_local_proj)
