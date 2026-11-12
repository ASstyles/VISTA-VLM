import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

class CLIPVisionTransformer(nn.Module):
    """
    CLIP ViT-B/16 Visual Encoder.
    Processes image crops (e.g. 224x224) and projects to normalized 512-dim embedding.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        out_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embed_dim))
        self.ln_pre = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Parameter(torch.randn(embed_dim, out_dim) * (embed_dim ** -0.5))

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Resize to 224x224 if necessary
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)

        B = x.shape[0]
        x = self.conv1(x) # [B, embed_dim, grid, grid]
        grid_h, grid_w = x.shape[-2:]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1) # [B, num_patches, embed_dim]

        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1) # [B, num_patches + 1, embed_dim]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)
        cls_out = self.ln_post(x[:, 0, :])
        feat = cls_out @ self.proj
        # L2 Normalize
        feat = F.normalize(feat, p=2, dim=-1)

        if not return_patches:
            return feat

        # Project spatial patch tokens: [B, num_patches, embed_dim] -> [B, num_patches, out_dim]
        patch_out = self.ln_post(x[:, 1:, :]) @ self.proj
        patch_out = F.normalize(patch_out, p=2, dim=-1)
        # Reshape to [B, out_dim, grid_h, grid_w]
        patch_grid = patch_out.permute(0, 2, 1).reshape(B, self.proj.shape[1], grid_h, grid_w)
        return feat, patch_grid


class SimpleCLIPTokenizer:
    """Lightweight self-contained character/subword tokenizer for industrial inspection prompts."""
    def __init__(self, context_length: int = 77):
        self.context_length = context_length
        self.vocab = {"<pad>": 0, "<start>": 1, "<end>": 2}
        # Prepopulate with ascii and common anomaly keywords
        chars = [chr(i) for i in range(32, 127)]
        for c in chars:
            self.vocab[c] = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens = [self.vocab["<start>"]]
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab["<pad>"])
        tokens.append(self.vocab["<end>"])
        # Pad or truncate
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length - 1] + [self.vocab["<end>"]]
        else:
            tokens = tokens + [self.vocab["<pad>"]] * (self.context_length - len(tokens))
        return tokens

    def __call__(self, texts: List[str]) -> torch.Tensor:
        tokenized = [self.encode(t) for t in texts]
        return torch.tensor(tokenized, dtype=torch.long)


class CLIPTextTransformer(nn.Module):
    """
    CLIP ViT-B/16 Text Encoder.
    Embeds text prompts and projects to normalized 512-dim embedding.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 512,
        out_dim: int = 512,
        context_length: int = 77,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.randn(embed_dim, out_dim) * (embed_dim ** -0.5))

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        B, N = text_tokens.shape
        x = self.token_embedding(text_tokens) + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        # Take the eot token (first token or max token)
        eot_idx = text_tokens.argmax(dim=-1)
        feat = x[torch.arange(B), eot_idx] @ self.text_projection
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


class CLIPModel(nn.Module):
    """Complete frozen CLIP model containing vision and text encoders."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.visual = CLIPVisionTransformer(embed_dim=768, out_dim=embed_dim)
        self.text = CLIPTextTransformer(embed_dim=embed_dim, out_dim=embed_dim)
        self.tokenizer = SimpleCLIPTokenizer()
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052) # ln(100)

    def encode_image(self, image: torch.Tensor, return_patches: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.visual(image, return_patches=return_patches)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        return self.text(tokens)

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between image and text features."""
        # Both image_features and text_features are already L2 normalized
        return image_features @ text_features.T
