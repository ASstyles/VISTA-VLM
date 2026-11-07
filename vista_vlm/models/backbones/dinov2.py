import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from vista_vlm.utils.registry import MODEL_REGISTRY

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x) # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2Backbone(nn.Module):
    """
    DINOv2 Vision Transformer Backbone for VISTA-VLM.
    Extracts multi-scale patch tokens and global CLS tokens from intermediate blocks.
    Default configuration:
      - Input resolution: 518 x 518
      - Patch size: 14 (37 x 37 spatial grid)
      - Blocks: 4, 8, 12 (1-based index)
      - ViT-S/14: embed_dim = 384
      - ViT-B/14: embed_dim = 768
    """
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        out_indices: List[int] = [4, 8, 12],
        pretrained: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_indices = [idx - 1 if idx > 0 else idx for idx in out_indices] # 0-based indexing
        self.grid_size = (img_size // patch_size, img_size // patch_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self._init_weights()
        if pretrained:
            self._load_pretrained()

    def _init_weights(self):
        # Truncated normal initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self):
        # Placeholder / hook for external weights loading if checkpoint provided
        pass

    def forward_features(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Extracts intermediate features.
        Returns:
            patch_features: List of tensors [B, embed_dim, grid_h, grid_w] for each out_index
            cls_features: List of tensors [B, embed_dim] for each out_index
            global_cls: Final normalized CLS token [B, embed_dim]
        """
        B = x.shape[0]
        x = self.patch_embed(x) # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, 1 + num_patches, embed_dim]
        x = x + self.pos_embed

        patch_features = []
        cls_features = []
        gh, gw = self.grid_size

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                cls_tok = x[:, 0] # [B, embed_dim]
                patch_tok = x[:, 1:] # [B, num_patches, embed_dim]
                # Reshape to spatial feature map: [B, embed_dim, gh, gw]
                spatial_feat = patch_tok.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)
                patch_features.append(spatial_feat)
                cls_features.append(cls_tok)

        final_norm = self.norm(x)
        global_cls = final_norm[:, 0]

        return {
            "patch_features": patch_features,
            "cls_features": cls_features,
            "global_cls": global_cls
        }

    def forward(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        return self.forward_features(x)


@MODEL_REGISTRY.register("dinov2_vits14")
def dinov2_vits14(img_size: int = 518, out_indices: List[int] = [4, 8, 12], pretrained: bool = False, **kwargs) -> DINOv2Backbone:
    return DINOv2Backbone(
        img_size=img_size,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        out_indices=out_indices,
        pretrained=pretrained,
        **kwargs
    )

@MODEL_REGISTRY.register("dinov2_vitb14")
def dinov2_vitb14(img_size: int = 518, out_indices: List[int] = [4, 8, 12], pretrained: bool = False, **kwargs) -> DINOv2Backbone:
    return DINOv2Backbone(
        img_size=img_size,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_indices=out_indices,
        pretrained=pretrained,
        **kwargs
    )
