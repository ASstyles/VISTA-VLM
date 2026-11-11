import os
import time
import base64
import io
import uuid
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Union, Optional, List

from vista_vlm.config.schema import VISTAConfig
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.models.language_verifier import LanguageVerifier
from vista_vlm.calibration.conformal import ConformalCalibrator
from vista_vlm.calibration.ood import OODDetector
from vista_vlm.calibration.selective import SelectivePredictor
from vista_vlm.data.transforms import VISTATransform
from vista_vlm.api.schemas import (
    InspectionResponse,
    InspectionStatus,
    BoundingBox,
    RegionFinding,
    CandidateRanking,
    ReliabilityIndicators
)
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.pipeline")

class VISTAInspectionPipeline:
    """
    End-to-End VISTA-VLM Unified Inspection Engine (Phase 10).
    Seamlessly orchestrates:
    - Deep spatial anomaly localization (DINOv2 + Global-Local Fusion + Conv Decoder)
    - Multimodal Vision-Language verification (frozen CLIP ViT-B/16 ROI crop interpretation)
    - Split conformal finite-sample calibration
    - Out-of-Distribution boundary detection
    - Selective prediction and human-in-the-loop escalation gating.
    """
    def __init__(
        self,
        config: Optional[VISTAConfig] = None,
        checkpoint_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        ood_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = config or VISTAConfig()
        self.device = device or self.config.system.device

        # 1. Initialize VISTA visual model
        self.model = VISTAAnomalyModel(
            backbone_name=self.config.model.backbone,
            embed_dim=self.config.model.embed_dim,
            out_indices=self.config.model.out_indices,
            proj_dim=self.config.model.fusion.proj_dim,
            image_size=self.config.data.image_size,
            fusion_type=self.config.model.fusion.type,
            use_film=self.config.model.fusion.use_film_conditioning,
            top_k_ratio=self.config.model.scoring.top_k_ratio,
            global_head_weight=self.config.model.scoring.global_head_weight,
            freeze_backbone=self.config.model.freeze_backbone,
        ).to(self.device)
        self.model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path, device=self.device)

        # 2. Initialize Language Verifier
        self.verifier = LanguageVerifier(
            embed_dim=self.config.language_verifier.embed_dim,
            roi_threshold=self.config.language_verifier.roi_threshold,
            min_box_area=self.config.language_verifier.min_box_area,
            box_padding=self.config.language_verifier.box_padding,
            ambiguity_margin=self.config.language_verifier.ambiguity_margin,
            min_confidence=self.config.language_verifier.min_confidence,
        ).to(self.device)
        self.verifier.eval()

        # 3. Initialize Calibration & Reliability modules
        self.calibrator = ConformalCalibrator(target_false_alarm_rate=self.config.calibration.target_false_alarm_rate)
        if calibration_path and os.path.exists(calibration_path):
            self.calibrator.load(calibration_path)

        self.ood_detector = OODDetector(
            metric=self.config.calibration.ood.metric,
            threshold_percentile=self.config.calibration.ood.threshold_percentile
        )
        if ood_path and os.path.exists(ood_path):
            self.ood_detector.load(ood_path)

        self.selective_predictor = SelectivePredictor(
            uncertainty_band_width=self.config.calibration.selective.uncertainty_band_width,
            enabled=self.config.calibration.selective.enable_human_escalation
        )

        # 4. Transform
        self.transform = VISTATransform(image_size=self.config.data.image_size, is_training=False)

    def _to_pil_image(self, image_input: Union[str, Image.Image, np.ndarray, bytes]) -> Image.Image:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _render_heatmap_base64(self, original_rgb: np.ndarray, anomaly_map: np.ndarray) -> str:
        """Renders color jet heatmap overlay on top of original image and converts to base64 PNG."""
        H, W = original_rgb.shape[:2]
        # Resize anomaly map to original dimensions if needed
        if anomaly_map.shape[:2] != (H, W):
            map_resized = cv2.resize(anomaly_map, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            map_resized = anomaly_map

        heatmap_uint8 = (np.clip(map_resized, 0, 1) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend 60% original + 40% heatmap
        blended = cv2.addWeighted(original_rgb, 0.6, heatmap_color_rgb, 0.4, 0)

        pil_img = Image.fromarray(blended)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _render_mask_base64(self, mask_uint8: np.ndarray) -> str:
        pil_img = Image.fromarray(mask_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def inspect(
        self,
        image_input: Union[str, Image.Image, np.ndarray, bytes],
        category: str = "bottle",
        custom_defect_candidates: Optional[List[str]] = None
    ) -> InspectionResponse:
        """
        Runs complete VISTA-VLM inspection pipeline on an image.
        """
        t0 = time.time()
        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"

        # 1. Image preprocessing
        pil_img = self._to_pil_image(image_input)
        orig_w, orig_h = pil_img.size
        orig_np = np.array(pil_img)

        img_tensor, _ = self.transform(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device) # [1, 3, 518, 518]

        # 2. VISTA model forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor)

        image_score = float(outputs["image_score"].squeeze().cpu().item())
        anomaly_map_tensor = outputs["anomaly_map"].squeeze().cpu().numpy() # [518, 518]
        global_features = outputs["global_features"].squeeze().cpu().numpy() # [embed_dim]

        # Resize anomaly map back to original input aspect ratio
        anomaly_map_orig = cv2.resize(anomaly_map_tensor, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 3. Calibration & Conformal Prediction
        is_calibrated = self.calibrator.is_calibrated(category)
        is_anomaly_calibrated, p_value, threshold = self.calibrator.predict(image_score, category=category, strict=False)

        # 4. Out-of-Distribution Detection
        is_ood, ood_dist, ood_thresh = self.ood_detector.compute_distance(global_features, category=category, strict=False)

        # 5. Multimodal Vision-Language Verification on ROIs
        rois = self.verifier.extract_rois(anomaly_map_orig, image_shape=(orig_h, orig_w))
        has_ambiguity = False

        if rois:
            region_findings_raw = self.verifier.verify_rois(
                image_np=orig_np,
                rois=rois,
                category=category,
                custom_candidates=custom_defect_candidates,
                device=self.device
            )
            for rf in region_findings_raw:
                if rf["is_ambiguous"]:
                    has_ambiguity = True
        else:
            region_findings_raw = []

        # 6. Selective Prediction & Human Escalation Evaluation under Fail-Safe Controls
        escalation_eval = self.selective_predictor.evaluate_decision(
            anomaly_score=image_score,
            calibrated_threshold=threshold,
            is_ood=is_ood,
            is_calibrated=is_calibrated,
            image_quality_ok=True,
            model_ready=True,
            is_region_ambiguous=has_ambiguity
        )

        # 7. Synthesize textual interpretation with evidence-grounded verifier
        if is_ood:
            status = InspectionStatus.OOD
        elif escalation_eval["human_review_required"]:
            status = InspectionStatus.ESCALATE_TO_HUMAN
        elif is_anomaly_calibrated:
            status = InspectionStatus.ANOMALY_DETECTED
        else:
            status = InspectionStatus.NORMAL

        text_summary = self.verifier.generate_grounded_explanation(
            is_anomaly=is_anomaly_calibrated,
            image_score=image_score,
            calibrated_threshold=threshold,
            findings=region_findings_raw,
            category=category,
            is_ood=is_ood,
            in_uncertainty_band=escalation_eval["in_uncertainty_band"]
        )

        # 8. Encode overlays
        heatmap_b64 = self._render_heatmap_base64(orig_np, anomaly_map_orig)
        binary_mask_np = (anomaly_map_orig > self.config.language_verifier.roi_threshold).astype(np.uint8) * 255
        mask_b64 = self._render_mask_base64(binary_mask_np)

        # 9. Format response objects
        bounding_boxes = [
            BoundingBox(
                x1=r["box"][0],
                y1=r["box"][1],
                x2=r["box"][2],
                y2=r["box"][3],
                area=r["area"],
                spatial_score=r["mean_score"]
            )
            for r in rois
        ]

        region_findings = [
            RegionFinding(
                box=rf["box"],
                area=rf["area"],
                spatial_anomaly_score=rf["spatial_anomaly_score"],
                top_defect_label=rf["top_defect_label"],
                top_similarity_score=rf["top_similarity_score"],
                confidence_margin=rf["confidence_margin"],
                is_ambiguous=rf["is_ambiguous"],
                candidate_rankings=[
                    CandidateRanking(
                        label=cr["label"],
                        score=cr["score"],
                        prompt=cr["prompt"]
                    )
                    for cr in rf["candidate_rankings"]
                ]
            )
            for rf in region_findings_raw
        ]

        reliability = ReliabilityIndicators(
            is_calibrated=is_calibrated,
            calibrated_threshold=round(threshold, 4) if threshold is not None else None,
            conformal_p_value=round(p_value, 4) if p_value is not None else None,
            target_false_alarm_rate=self.config.calibration.target_false_alarm_rate,
            is_ood=is_ood,
            ood_distance=ood_dist,
            ood_threshold=ood_thresh,
            in_uncertainty_band=escalation_eval["in_uncertainty_band"],
            operational_state=escalation_eval["state"],
            disposition=escalation_eval["disposition"],
            human_review_recommended=escalation_eval["human_review_required"],
            review_reason=escalation_eval["reason"]
        )

        elapsed_ms = round((time.time() - t0) * 1000.0, 2)

        return InspectionResponse(
            inspection_id=inspection_id,
            timestamp=datetime.now().isoformat(),
            category=category,
            status=status,
            image_anomaly_score=round(image_score, 4),
            is_anomaly=is_anomaly_calibrated,
            threshold=round(threshold, 4) if threshold is not None else 0.5,
            num_defects_found=len(bounding_boxes),
            bounding_boxes=bounding_boxes,
            region_findings=region_findings,
            textual_interpretation=text_summary,
            reliability=reliability,
            heatmap_base64=heatmap_b64,
            mask_base64=mask_b64,
            execution_time_ms=elapsed_ms
        )
