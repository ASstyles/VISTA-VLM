from typing import Dict, Any, Tuple, Optional

class SelectivePredictor:
    """
    Selective Prediction & Escalation Engine (Phase 11).
    Determines whether a prediction can be safely committed autonomously or requires
    human-in-the-loop review based on:
    1. Conformal uncertainty band [tau_c - delta, tau_c + delta]
    2. Out-of-Distribution status
    3. Vision-language semantic ambiguity
    """
    def __init__(self, uncertainty_band_width: float = 0.08, enabled: bool = True):
        self.uncertainty_band = uncertainty_band_width
        self.enabled = enabled

    def evaluate_decision(
        self,
        anomaly_score: float,
        calibrated_threshold: float,
        is_ood: bool,
        is_region_ambiguous: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates prediction confidence and human review recommendation.
        """
        if not self.enabled:
            is_anomaly = anomaly_score > calibrated_threshold
            return {
                "autonomous_action": "ANOMALY_DETECTED" if is_anomaly else "NORMAL",
                "human_review_required": False,
                "reason": "Autonomous prediction enabled without selective gating",
                "in_uncertainty_band": False
            }

        # 1. Check OOD
        if is_ood:
            return {
                "autonomous_action": "ESCALATE_TO_HUMAN",
                "human_review_required": True,
                "reason": "Out-of-Distribution (OOD) visual signature detected; object differs from known normal distribution.",
                "in_uncertainty_band": False
            }

        # 2. Check conformal uncertainty band
        low_bound = max(0.0, calibrated_threshold - self.uncertainty_band)
        high_bound = min(1.0, calibrated_threshold + self.uncertainty_band)
        in_band = (low_bound <= anomaly_score <= high_bound)

        if in_band:
            return {
                "autonomous_action": "ESCALATE_TO_HUMAN",
                "human_review_required": True,
                "reason": f"Anomaly score ({anomaly_score:.3f}) falls within conformal uncertainty band [{low_bound:.3f}, {high_bound:.3f}].",
                "in_uncertainty_band": True
            }

        # 3. Check language ambiguity
        if is_region_ambiguous and (anomaly_score > calibrated_threshold):
            return {
                "autonomous_action": "ESCALATE_TO_HUMAN",
                "human_review_required": True,
                "reason": "Visual defect detected, but semantic defect classification is ambiguous or unidentified.",
                "in_uncertainty_band": False
            }

        # 4. Confident autonomous decision
        is_anomaly = anomaly_score > calibrated_threshold
        action = "ANOMALY_DETECTED" if is_anomaly else "NORMAL"
        return {
            "autonomous_action": action,
            "human_review_required": False,
            "reason": f"High-confidence autonomous decision (score {anomaly_score:.3f} vs threshold {calibrated_threshold:.3f}).",
            "in_uncertainty_band": False
        }
