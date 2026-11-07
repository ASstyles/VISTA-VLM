import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple

class ConformalCalibrator:
    """
    Split Conformal Prediction Framework for Industrial Anomaly Detection (Phase 11).
    Provides finite-sample false-alarm guarantees under exchangeability.
    Calibrates non-conformity thresholds using held-out normal images.
    """
    def __init__(self, target_false_alarm_rate: float = 0.05):
        self.alpha = target_false_alarm_rate # e.g. 0.05 for 95% specificity
        self.category_thresholds: Dict[str, float] = {}
        self.calibration_scores: Dict[str, List[float]] = {}
        self.sample_counts: Dict[str, int] = {}

    def fit(self, scores: List[float], category: str = "default") -> float:
        """
        Fits conformal quantile threshold on independent normal calibration scores.
        Args:
            scores: Anomaly scores on held-out normal images
            category: Category name
        Returns:
            calibrated_threshold tau_c
        """
        n = len(scores)
        if n == 0:
            raise ValueError(f"Cannot calibrate with 0 samples for category {category}")

        scores_arr = np.sort(np.array(scores, dtype=np.float64))
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        threshold = float(np.quantile(scores_arr, q_level, method="higher"))

        self.category_thresholds[category] = threshold
        self.calibration_scores[category] = scores_arr.tolist()
        self.sample_counts[category] = n

        return threshold

    def predict(self, score: float, category: str = "default") -> Tuple[bool, float, float]:
        """
        Evaluates test score against calibrated conformal threshold.
        Returns:
            is_anomaly: bool (True if score > threshold)
            conformal_p_value: float
            threshold: float
        """
        threshold = self.category_thresholds.get(category, 0.5)
        cal_scores = self.calibration_scores.get(category, [])
        n = len(cal_scores)

        if n > 0:
            # Empirical p-value: (1 + sum(s_cal >= s_test)) / (n + 1)
            # Under null (normal), P(p_val <= alpha) <= alpha
            count_greater = np.sum(np.array(cal_scores) >= score)
            p_val = float((1.0 + count_greater) / (n + 1.0))
        else:
            p_val = 1.0 - score # Heuristic fallback if uncalibrated

        is_anomaly = bool(score > threshold)
        return is_anomaly, p_val, threshold

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "alpha": self.alpha,
            "category_thresholds": self.category_thresholds,
            "calibration_scores": self.calibration_scores,
            "sample_counts": self.sample_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.alpha = data["alpha"]
        self.category_thresholds = data["category_thresholds"]
        self.calibration_scores = data["calibration_scores"]
        self.sample_counts = data["sample_counts"]
