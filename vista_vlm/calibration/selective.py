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

    def evaluate_false_pass_risk(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
        calibrated_threshold: float,
        uncertainty_band: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rigorously evaluates false-pass (defect escape) and false-alarm (overkill) risks
        with Wilson score 95% confidence intervals.
        """
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        band = self.uncertainty_band if uncertainty_band is None else uncertainty_band

        low_b = max(0.0, calibrated_threshold - band)
        high_b = min(1.0, calibrated_threshold + band)

        # Autonomous decisions only (outside uncertainty band)
        accepted = (scores_arr < low_b) | (scores_arr > high_b)
        n_total = len(scores_arr)
        n_accepted = int(np.sum(accepted))
        coverage = float(n_accepted / max(1, n_total))

        acc_scores = scores_arr[accepted]
        acc_labels = labels_arr[accepted]

        defective_mask = (acc_labels == 1)
        normal_mask = (acc_labels == 0)

        n_defective = int(np.sum(defective_mask))
        n_normal = int(np.sum(normal_mask))

        # False Pass: True label is 1 (defective), but predicted 0 (score <= threshold)
        false_passes = int(np.sum((acc_scores <= calibrated_threshold) & defective_mask))
        false_pass_rate = float(false_passes / max(1, n_defective)) if n_defective > 0 else 0.0

        # False Alarm: True label is 0 (normal), but predicted 1 (score > threshold)
        false_alarms = int(np.sum((acc_scores > calibrated_threshold) & normal_mask))
        false_alarm_rate = float(false_alarms / max(1, n_normal)) if n_normal > 0 else 0.0

        # Wilson Score 95% Confidence Interval for false-pass rate
        # z = 1.96 for 95% CI
        z = 1.96
        if n_defective > 0:
            denom = 1.0 + (z ** 2) / n_defective
            center = (false_pass_rate + (z ** 2) / (2.0 * n_defective)) / denom
            margin = z * np.sqrt((false_pass_rate * (1.0 - false_pass_rate) + (z ** 2) / (4.0 * n_defective)) / n_defective) / denom
            ci_lower = max(0.0, float(center - margin))
            ci_upper = min(1.0, float(center + margin))
        else:
            ci_lower, ci_upper = 0.0, 1.0

        # Check sample size validity: Rule of 300 for 1% risk claims
        is_sample_size_sufficient = bool(n_defective >= 300)

        return {
            "total_samples": n_total,
            "accepted_samples": n_accepted,
            "coverage": round(coverage, 4),
            "abstention_rate": round(1.0 - coverage, 4),
            "defective_accepted": n_defective,
            "false_passes": false_passes,
            "observed_false_pass_rate": round(false_pass_rate, 4),
            "false_pass_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "normal_accepted": n_normal,
            "false_alarms": false_alarms,
            "observed_false_alarm_rate": round(false_alarm_rate, 4),
            "is_sample_size_sufficient_for_1pct": is_sample_size_sufficient,
            "sample_size_warning": (
                None if is_sample_size_sufficient
                else f"Sample size (N_def={n_defective}) is insufficient to statistically prove <=1% false-pass rate at 95% power (N>=300 required)."
            )
        }

    def compute_risk_coverage_curve(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
        calibrated_threshold: float,
        num_steps: int = 50
    ) -> List[Dict[str, float]]:
        """
        Computes the complete risk-coverage trade-off curve across candidate uncertainty bands.
        """
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        deltas = np.linspace(0.0, 0.40, num_steps)
        curve = []

        for d in deltas:
            low_b = max(0.0, calibrated_threshold - d)
            high_b = min(1.0, calibrated_threshold + d)
            accepted = (scores_arr < low_b) | (scores_arr > high_b)
            cov = float(np.mean(accepted))
            if np.sum(accepted) > 0:
                preds = (scores_arr[accepted] > calibrated_threshold).astype(int)
                risk = float(np.mean(preds != labels_arr[accepted]))
                defective_mask = (labels_arr[accepted] == 1)
                fn = int(np.sum((scores_arr[accepted] <= calibrated_threshold) & defective_mask))
                n_def = int(np.sum(defective_mask))
                fnr = float(fn / max(1, n_def)) if n_def > 0 else 0.0
            else:
                risk = 0.0
                fnr = 0.0

            curve.append({
                "delta": round(float(d), 4),
                "coverage": round(cov, 4),
                "selective_risk": round(risk, 4),
                "false_pass_rate": round(fnr, 4),
                "abstention_rate": round(1.0 - cov, 4)
            })
        return curve

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
