import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from vista_vlm.models.backbones.clip_vision import CLIPModel
from vista_vlm.utils.registry import BASELINE_REGISTRY

@BASELINE_REGISTRY.register("winclip")
class WinCLIP(nn.Module):
    """
    WinCLIP: Window-based CLIP for Zero-Shot and Few-Shot Anomaly Detection (Phase 4).
    Uses compositional prompt ensembles and multi-scale visual-language patch alignment.
    """
    def __init__(self, embed_dim: int = 512, image_size: int = 518):
        super().__init__()
        self.clip = CLIPModel(embed_dim=embed_dim)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.image_size = image_size

    def get_prompt_ensembles(self, category: str) -> Tuple[List[str], List[str]]:
        normal_prompts = [
            f"a photo of a flawless {category}",
            f"a photo of a perfect, undamaged {category}",
            f"a photo of a clean, normal {category}",
            f"a crop of a defect-free {category}"
        ]
        anomaly_prompts = [
            f"a photo of a damaged {category}",
            f"a photo of a {category} with scratches, cracks, or flaws",
            f"a photo of a defective {category}",
            f"a crop of an anomalous {category}"
        ]
        return normal_prompts, anomaly_prompts

    def predict(self, x: torch.Tensor, category: str = "bottle") -> Dict[str, torch.Tensor]:
        device = x.device
        self.clip.to(device)

        normal_prompts, anomaly_prompts = self.get_prompt_ensembles(category)

        with torch.no_grad():
            norm_feats = self.clip.encode_text(normal_prompts) # [N_norm, 512]
            anom_feats = self.clip.encode_text(anomaly_prompts) # [N_anom, 512]

            # Mean prompt embeddings
            norm_text = F.normalize(norm_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
            anom_text = F.normalize(anom_feats.mean(dim=0, keepdim=True), p=2, dim=-1)

            # Image global features and spatial patch features (14x14 grid)
            img_feats, patch_grid = self.clip.encode_image(x, return_patches=True) # [B, 512], [B, 512, G, G]

            # Cosine similarity with normal vs anomaly
            sim_norm = img_feats @ norm_text.T # [B, 1]
            sim_anom = img_feats @ anom_text.T # [B, 1]

            # Softmax probability for global anomaly class
            logits = torch.cat([sim_norm, sim_anom], dim=1) * 10.0 # [B, 2]
            probs = F.softmax(logits, dim=1)
            global_score = probs[:, 1] # [B]

            # Patch-level dense visual-textual spatial alignment (Jeong et al., WinCLIP CVPR 2023)
            sim_patch_norm = (patch_grid * norm_text.view(1, -1, 1, 1)).sum(dim=1, keepdim=True) # [B, 1, G, G]
            sim_patch_anom = (patch_grid * anom_text.view(1, -1, 1, 1)).sum(dim=1, keepdim=True) # [B, 1, G, G]
            patch_logits = torch.cat([sim_patch_norm, sim_patch_anom], dim=1) * 10.0
            patch_probs = F.softmax(patch_logits, dim=1)
            raw_patch_map = patch_probs[:, 1:2, :, :] # [B, 1, G, G]

            # Upsample patch anomaly map to image resolution
            anomaly_map = F.interpolate(raw_patch_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            # Combined WinCLIP image score (global visual-text + top patch anomaly)
            B = x.shape[0]
            max_patch_score = anomaly_map.view(B, -1).max(dim=1)[0]
            image_score = 0.5 * global_score + 0.5 * max_patch_score

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
