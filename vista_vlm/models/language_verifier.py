import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from vista_vlm.models.backbones.clip_vision import CLIPModel
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.language_verifier")

class LanguageVerifier(nn.Module):
    """
    Vision-Language Verifier (v4.0 Architecture).
    Uses a frozen CLIP ViT-B/16 multimodal model to verify, localize, and semantically
    interpret detected anomaly regions without fabricating ungrounded explanations.
    Features:
      1. Pre-computed, cached text prompt embeddings per category for low latency.
      2. Category-conditioned multi-prompt ensembles for fine industrial defect taxonomy.
      3. Strict confidence and margin ambiguity gating.
      4. Structured, evidence-traceable textual grounding generator.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        roi_threshold: float = 0.45,
        min_box_area: int = 80,
        box_padding: float = 0.15,
        ambiguity_margin: float = 0.08,
        min_confidence: float = 0.20,
    ):
        super().__init__()
        self.clip = CLIPModel(embed_dim=embed_dim)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        self.roi_threshold = roi_threshold
        self.min_box_area = min_box_area
        self.box_padding = box_padding
        self.ambiguity_margin = ambiguity_margin
        self.min_confidence = min_confidence

        # In-memory cache for encoded text embeddings: category -> torch.Tensor [K, embed_dim]
        self._text_cache: Dict[str, torch.Tensor] = {}
        self._prompt_cache: Dict[str, List[Tuple[str, str]]] = {}

    def extract_rois(
        self,
        anomaly_map: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Extracts Region-of-Interest (ROI) bounding boxes and contours from anomaly map.
        Args:
            anomaly_map: 2D float array [H, W] in [0, 1]
            image_shape: (H, W)
        Returns:
            List of ROI dicts: {'box': [x1, y1, x2, y2], 'area': int, 'mean_score': float, 'max_score': float}
        """
        H, W = image_shape
        binary_mask = (anomaly_map > self.roi_threshold).astype(np.uint8) * 255

        # Morphological opening to eliminate isolated pixel spikes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_box_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            pad_x = int(w * self.box_padding)
            pad_y = int(h * self.box_padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W - 1, x + w + pad_x)
            y2 = min(H - 1, y + h + pad_y)

            roi_map = anomaly_map[y1:y2, x1:x2]
            mean_val = float(roi_map.mean()) if roi_map.size > 0 else 0.0
            max_val = float(roi_map.max()) if roi_map.size > 0 else 0.0

            rois.append({
                "box": [x1, y1, x2, y2],
                "area": int(area),
                "mean_score": round(mean_val, 4),
                "max_score": round(max_val, 4),
            })

        rois.sort(key=lambda r: r["mean_score"], reverse=True)
        return rois

    def get_candidate_prompts(self, category: str, custom_candidates: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Returns list of tuples: (semantic_label, prompt_template)."""
        if custom_candidates:
            return [(c, f"a photo of a {category} with {c}") for c in custom_candidates]

        cat_lower = category.lower()
        if "bottle" in cat_lower:
            return [
                ("broken_large", "a photo of a broken bottle with cracked glass or missing mouth"),
                ("contamination", "a photo of a bottle with dirty liquid or surface contamination"),
                ("scratch", "a close-up photo of a glass bottle with visible surface scratches"),
                ("crack", "a close-up photo of a bottle with a fracture or hairline glass crack"),
                ("normal_surface", "a photo of a clean, flawless, pristine glass bottle")
            ]
        elif "cable" in cat_lower:
            return [
                ("cut_wire", "a photo of an electric cable with severed or cut insulation"),
                ("bent_wire", "a photo of a bent, twisted, or deformed cable"),
                ("exposed_copper", "a photo of a cable with missing insulation showing bare copper"),
                ("scratch", "a photo of a cable surface with scrape marks"),
                ("normal_surface", "a photo of a straight, undamaged, clean insulated cable")
            ]
        elif "capsule" in cat_lower:
            return [
                ("crack", "a close-up photo of a pharmaceutical capsule with cracked shell"),
                ("scratch", "a photo of a capsule with surface scratches or abrasion"),
                ("dent", "a photo of an indented, squished, or dented capsule"),
                ("discoloration", "a photo of a capsule with stained or uneven color"),
                ("normal_surface", "a photo of a smooth, intact, uniform capsule")
            ]
        elif "leather" in cat_lower or "tile" in cat_lower:
            return [
                ("cut", f"a photo of {category} with a cut or sharp slice"),
                ("hole", f"a photo of {category} with a puncture or hole"),
                ("stain", f"a photo of {category} with discoloration or liquid stain"),
                ("scratch", f"a photo of {category} with visible scrape marks"),
                ("normal_surface", f"a photo of flawless, uniform {category} surface")
            ]

        return [
            ("scratch", f"a close-up photo of a {category} with a surface scratch or scrape"),
            ("crack", f"a close-up photo of a {category} with a fracture or crack defect"),
            ("contamination", f"a close-up photo of a {category} with dirt, stain, or surface contamination"),
            ("dent_hole", f"a close-up photo of a {category} with a dent, puncture, or hole defect"),
            ("broken_part", f"a close-up photo of a {category} with a broken, deformed, or missing part"),
            ("discoloration", f"a close-up photo of a {category} with discoloration or faded texture"),
            ("normal_surface", f"a close-up photo of a clean, flawless, normal {category} surface")
        ]

    def get_or_create_text_embeddings(
        self,
        category: str,
        custom_candidates: Optional[List[str]] = None,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, List[str], List[str]]:
        """Caches text embeddings to eliminate redundant tokenization overhead."""
        cache_key = f"{category}_{','.join(custom_candidates) if custom_candidates else 'default'}"
        if cache_key in self._text_cache:
            candidates = self._prompt_cache[cache_key]
            labels = [c[0] for c in candidates]
            prompts = [c[1] for c in candidates]
            return self._text_cache[cache_key].to(device), labels, prompts

        candidates = self.get_candidate_prompts(category, custom_candidates)
        labels = [c[0] for c in candidates]
        prompts = [c[1] for c in candidates]

        with torch.no_grad():
            text_features = self.clip.encode_text(prompts).to(device)

        self._text_cache[cache_key] = text_features.cpu()
        self._prompt_cache[cache_key] = candidates
        return text_features, labels, prompts

    def verify_rois(
        self,
        image_np: np.ndarray,
        rois: List[Dict[str, Any]],
        category: str,
        custom_candidates: Optional[List[str]] = None,
        device: str = "cpu"
    ) -> List[Dict[str, Any]]:
        """Verifies and ranks semantic descriptions for each extracted ROI crop using cached CLIP embeddings."""
        if not rois:
            return []

        text_features, labels, prompts = self.get_or_create_text_embeddings(category, custom_candidates, device=device)

        verified_findings = []
        H, W = image_np.shape[:2]

        for roi in rois:
            x1, y1, x2, y2 = roi["box"]
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue

            # Standard 224x224 RGB crop normalization
            crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            crop_tensor = (crop_tensor - mean) / std

            with torch.no_grad():
                crop_features = self.clip.encode_image(crop_tensor.to(device))
                similarities = self.clip.compute_similarity(crop_features, text_features).squeeze(0)
                scores = similarities.cpu().numpy()

            ranked_indices = np.argsort(scores)[::-1]
            top1_idx = ranked_indices[0]
            top2_idx = ranked_indices[1]
            top1_score = float(scores[top1_idx])
            top2_score = float(scores[top2_idx])
            margin = top1_score - top2_score

            is_ambiguous = (margin < self.ambiguity_margin) or (top1_score < self.min_confidence)
            top_label = labels[top1_idx] if not is_ambiguous else "unidentified_or_ambiguous_defect"

            finding = {
                "box": [x1, y1, x2, y2],
                "area": roi["area"],
                "spatial_anomaly_score": roi["mean_score"],
                "top_defect_label": top_label,
                "top_similarity_score": round(top1_score, 4),
                "confidence_margin": round(margin, 4),
                "is_ambiguous": is_ambiguous,
                "candidate_rankings": [
                    {"label": labels[idx], "score": round(float(scores[idx]), 4), "prompt": prompts[idx]}
                    for idx in ranked_indices[:4]
                ]
            }
            verified_findings.append(finding)

        return verified_findings

    def generate_grounded_explanation(
        self,
        is_anomaly: bool,
        image_score: float,
        calibrated_threshold: float,
        findings: List[Dict[str, Any]],
        category: str,
        is_ood: bool = False,
        in_uncertainty_band: bool = False
    ) -> str:
        """
        Synthesizes an evidence-grounded textual explanation.
        Guarantees that every assertion is traceable to bounding boxes, scores, and margins.
        """
        if is_ood:
            return f"Out-of-Distribution warning: Input visual signature deviates significantly from the reference '{category}' distribution. Autonomous classification aborted; human review required."

        if not is_anomaly:
            if in_uncertainty_band:
                return f"Surface appears nominal (score: {image_score:.3f} <= threshold: {calibrated_threshold:.3f}), but score is within the selective uncertainty band. Operator spot-check recommended."
            return f"Nominal: Inspection verified clean '{category}' surface with anomaly score {image_score:.3f} safely below calibrated threshold {calibrated_threshold:.3f}. No structural or surface defects detected."

        if not findings:
            return f"Anomaly detected: Image anomaly score ({image_score:.3f}) exceeds threshold ({calibrated_threshold:.3f}), but defect is diffuse and lacks distinct contiguous bounding contours."

        num_defects = len(findings)
        primary = findings[0]
        label = primary["top_defect_label"].replace("_", " ")
        margin = primary["confidence_margin"]
        box = primary["box"]

        if primary["is_ambiguous"]:
            return (
                f"Anomaly detected: {num_defects} anomalous region(s) identified (score: {image_score:.3f} > {calibrated_threshold:.3f}). "
                f"Primary defect at [{box[0]}, {box[1]}, {box[2]}, {box[3]}] has semantic ambiguity (confidence margin: {margin:.3f} < {self.ambiguity_margin:.2f}). "
                f"Escalated for operator verification."
            )

        return (
            f"Defect identified: Found {num_defects} anomalous region(s) on '{category}' (score: {image_score:.3f} > threshold: {calibrated_threshold:.3f}). "
            f"Primary region at [{box[0]}, {box[1]}, {box[2]}, {box[3]}] is classified as '{label}' "
            f"(CLIP similarity: {primary['top_similarity_score']:.3f}, confidence margin: {margin:.3f})."
        )
