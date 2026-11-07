import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
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

    def get_agnostic_prompts(self) -> Tuple[List[str], List[str]]:
        normal_prompts = [
            "a photo of a normal object without flaws",
            "a close-up of a flawless, pristine surface"
        ]
        anomaly_prompts = [
            "a photo of a damaged object with defects",
            "a close-up of a damaged, scratched, or cracked surface"
        ]
        return normal_prompts, anomaly_prompts

    def predict(self, x: torch.Tensor, category: str = "object") -> Dict[str, torch.Tensor]:
        device = x.device
        self.clip.to(device)

        normal_prompts, anomaly_prompts = self.get_agnostic_prompts()

        with torch.no_grad():
            norm_feats = self.clip.encode_text(normal_prompts)
            anom_feats = self.clip.encode_text(anomaly_prompts)

            norm_text = F.normalize(norm_feats.mean(dim=0, keepdim=True), p=2, dim=-1)
            anom_text = F.normalize(anom_feats.mean(dim=0, keepdim=True), p=2, dim=-1)

            img_feats = self.clip.encode_image(x)

            sim_norm = img_feats @ norm_text.T
            sim_anom = img_feats @ anom_text.T

            logits = torch.cat([sim_norm, sim_anom], dim=1) * 10.0
            probs = F.softmax(logits, dim=1)
            image_score = probs[:, 1]

            B = x.shape[0]
            anomaly_map = image_score.view(B, 1, 1, 1).expand(B, 1, self.image_size, self.image_size)

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
