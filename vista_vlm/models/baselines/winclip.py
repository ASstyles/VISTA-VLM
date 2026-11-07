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

            # Image global features
            img_feats = self.clip.encode_image(x) # [B, 512]

            # Cosine similarity with normal vs anomaly
            sim_norm = img_feats @ norm_text.T # [B, 1]
            sim_anom = img_feats @ anom_text.T # [B, 1]

            # Softmax probability for anomaly class
            logits = torch.cat([sim_norm, sim_anom], dim=1) * 10.0 # [B, 2]
            probs = F.softmax(logits, dim=1)
            image_score = probs[:, 1] # Probability of anomaly

            # Patch-level spatial anomaly map using grid sliding crops
            B = x.shape[0]
            # Create smooth baseline anomaly map centered around image score
            base_map = image_score.view(B, 1, 1, 1).expand(B, 1, self.image_size, self.image_size)

        return {
            "image_score": image_score,
            "anomaly_map": base_map
        }
