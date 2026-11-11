import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple


class UncalibratedError(ValueError):
    """Raised when an anomaly score is evaluated against an uncalibrated category."""
    pass


class ConformalCalibrator:
    """
    Split Conformal Prediction Framework for Industrial Anomaly Detection (Phase 7 & Phase 11).
    Provides finite-sample false-alarm guarantees under exchangeability.
    Strictly forbids silent heuristic fallback thresholds or fabricated p-values.
    """
    def __init__(self, target_false_alarm_rate: float = 0.05):
        self.alpha = target_false_alarm_rate  # e.g. 0.05 for 95% specificity
        self.category_thresholds: Dict[str, float] = {}
        self.calibration_scores: Dict[str, List[float]] = {}
        self.sample_counts: Dict[str, int] = {}

    def is_calibrated(self, category: str = "default") -> bool:
        """Checks if category has valid calibrated threshold and empirical holdout scores."""
        return (
            category in self.category_thresholds
            and category in self.calibration_scores
            and len(self.calibration_scores[category]) > 0
        )

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
            raise ValueError(f"Cannot calibrate with 0 samples for category '{category}'")

        scores_arr = np.sort(np.array(scores, dtype=np.float64))
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        threshold = float(np.quantile(scores_arr, q_level, method="higher"))

        self.category_thresholds[category] = threshold
        self.calibration_scores[category] = scores_arr.tolist()
        self.sample_counts[category] = n

        return threshold

    def predict(
        self,
        score: float,
        category: str = "default",
        strict: bool = True
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Evaluates test score against calibrated conformal threshold.
        When uncalibrated, strictly refuses to substitute arbitrary heuristic numbers.
        Args:
            score: Continuous anomaly score in [0, 1]
            category: Category name
            strict: If True, raises UncalibratedError when category is uncalibrated.
                   If False, returns (False, None, None) signaling uncalibrated state.
        Returns:
            is_anomaly: bool
            conformal_p_value: Optional[float]
            threshold: Optional[float]
        """
        if not self.is_calibrated(category):
            if strict:
                raise UncalibratedError(
                    f"UNCALIBRATED: Category '{category}' has no valid split conformal calibration. "
                    f"Silent heuristic fallbacks are disabled under safety controls."
                )
            return False, None, None

        threshold = self.category_thresholds[category]
        cal_scores = self.calibration_scores[category]
        n = len(cal_scores)

        # Empirical non-conformity p-value under exchangeability:
        # P_val = (1 + sum(s_cal >= s_test)) / (n + 1)
        count_greater = np.sum(np.array(cal_scores) >= score)
        p_val = float((1.0 + count_greater) / (n + 1.0))

        is_anomaly = bool(score > threshold)
        return is_anomaly, round(p_val, 6), round(threshold, 6)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "5.0.0",
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
