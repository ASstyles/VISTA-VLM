import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from vista_vlm.models.backbones.clip_vision import CLIPModel
from vista_vlm.utils.registry import BASELINE_REGISTRY

@BASELINE_REGISTRY.register("anomalyclip")
class AnomalyCLIP(nn.Module):
    """
    AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection (Phase 4).
    Uses object-agnostic textual cues and token-level spatial alignment.
    """
    def __init__(self, embed_dim: int = 512, image_size: int = 518):
        super().__init__()
        self.clip = CLIPModel(embed_dim=embed_dim)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.image_size = image_size

    def get_agnostic_prompts(self, category: str = "object") -> Tuple[List[str], List[str]]:
        normal_prompts = [
            f"a photo of a normal {category} without flaws",
            f"a close-up of a flawless, pristine {category} surface",
            f"a clean, defect-free {category}"
        ]
        anomaly_prompts = [
            f"a photo of a damaged {category} with defects",
            f"a close-up of a damaged, scratched, or cracked {category} surface",
            f"a defective {category} with anomalies"
        ]
        return normal_prompts, anomaly_prompts

    def predict(self, x: torch.Tensor, category: str = "object") -> Dict[str, torch.Tensor]:
        device = x.device
        self.clip.to(device)

        normal_prompts, anomaly_prompts = self.get_agnostic_prompts(category)

        with torch.no_grad():
            norm_feats = self.clip.encode_text(normal_prompts)
            anom_feats = self.clip.encode_text(anomaly_prompts)

            norm_text = F.normalize(norm_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
            anom_text = F.normalize(anom_feats.mean(dim=0, keepdim=True), p=2, dim=-1)

            # Extract global features and spatial patch features (14x14 grid)
            img_feats, patch_grid = self.clip.encode_image(x, return_patches=True) # [B, 512], [B, 512, G, G]

            sim_norm = img_feats @ norm_text.T
            sim_anom = img_feats @ anom_text.T

            logits = torch.cat([sim_norm, sim_anom], dim=1) * 10.0
            probs = F.softmax(logits, dim=1)
            global_score = probs[:, 1]

            # Patch-level dense visual-textual spatial alignment
            sim_patch_norm = (patch_grid * norm_text.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            sim_patch_anom = (patch_grid * anom_text.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            patch_logits = torch.cat([sim_patch_norm, sim_patch_anom], dim=1) * 10.0
            patch_probs = F.softmax(patch_logits, dim=1)
            raw_patch_map = patch_probs[:, 1:2, :, :] # [B, 1, G, G]

            # Bilinear upsampling to original image size
            anomaly_map = F.interpolate(raw_patch_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            B = x.shape[0]
            max_patch_score = anomaly_map.view(B, -1).max(dim=1)[0]
            image_score = 0.5 * global_score + 0.5 * max_patch_score

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
