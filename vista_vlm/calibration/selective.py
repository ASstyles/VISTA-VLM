import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List

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

    def calibrate_uncertainty_band(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
        calibrated_threshold: float,
        target_risk: float = 0.01,
        min_coverage: float = 0.50
    ) -> float:
        """
        Calibrates empirical uncertainty band delta on validation samples to guarantee
        selective risk (error rate on autonomously accepted decisions) <= target_risk.
        """
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)

        best_delta = self.uncertainty_band
        candidate_deltas = np.linspace(0.01, 0.25, 50)

        for delta in candidate_deltas:
            low_b = max(0.0, calibrated_threshold - delta)
            high_b = min(1.0, calibrated_threshold + delta)
            accepted = (scores_arr < low_b) | (scores_arr > high_b)

            if np.mean(accepted) < min_coverage:
                continue

            pred_accepted = (scores_arr[accepted] > calibrated_threshold).astype(int)
            true_accepted = labels_arr[accepted]
            risk = float(np.mean(pred_accepted != true_accepted)) if len(pred_accepted) > 0 else 0.0

            if risk <= target_risk:
                best_delta = float(delta)
                break

        self.uncertainty_band = round(best_delta, 4)
        return self.uncertainty_band

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
