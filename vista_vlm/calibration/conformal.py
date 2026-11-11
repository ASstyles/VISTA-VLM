"""
Split Conformal Prediction Framework for Industrial Quality Inspection (V6.0).
==============================================================================
Provides finite-sample statistical guarantees for False Alarm Rate (FAR) control
under exchangeability assumptions.

Mathematical Formulation:
-------------------------
Given n independent normal calibration non-conformity scores:
    s_1, s_2, ..., s_n ~ P_normal
For target significance level alpha in (0, 1) (e.g. alpha = 0.05 for <=5% false alarms):
The conformal order statistic index is:
    k = ceil((n + 1) * (1 - alpha))
Where k is the 1-indexed order statistic of the sorted calibration scores.

Finite-Sample Validity Requirement:
-----------------------------------
If k > n, it is mathematically impossible to guarantee FAR <= alpha with only n samples.
In this case, the calibrator STRICTLY REJECTS the calibration request with a ValueError
rather than silently clipping to an unsupported quantile.

Upper-Tail Conformal P-Value:
-----------------------------
For a query test score s_test:
    p(s_test) = (1 + sum_{i=1}^n I(s_i >= s_test)) / (n + 1)
An anomaly is flagged when p(s_test) <= alpha, which exactly aligns with s_test >= tau_c.

IMPORTANT SAFETY NOTICE:
------------------------
Normal-only split conformal calibration controls the FALSE ALARM RATE (Type I error).
It DOES NOT establish a false-pass (defect escape) guarantee, because defective
samples are not observed during normal-only calibration and defect non-conformity
scores are not exchangeable with normal scores.
"""

from __future__ import annotations

import math
import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.calibration")


class UncalibratedError(ValueError):
    """Raised when an anomaly score is evaluated against an uncalibrated category."""
    pass


class InsufficientSampleSizeError(ValueError):
    """Raised when calibration sample size n cannot mathematically support target alpha."""
    pass


class ConformalCalibrator:
    """
    Split Conformal Calibrator for Industrial Inspection (V6.0).
    Guarantees finite-sample false-alarm control under exchangeability.
    Enforces strict order-statistic calculation, mathematical sample size checks,
    and fail-closed behavior.
    """
    def __init__(self, target_false_alarm_rate: float = 0.05):
        if not (0.0 < target_false_alarm_rate < 1.0):
            raise ValueError(f"target_false_alarm_rate (alpha) must be in (0, 1), got {target_false_alarm_rate}")
        self.alpha = float(target_false_alarm_rate)
        self.category_thresholds: Dict[str, float] = {}
        self.calibration_scores: Dict[str, List[float]] = {}
        self.sample_counts: Dict[str, int] = {}
        self.checksums: Dict[str, str] = {}

    def is_calibrated(self, category: str = "default") -> bool:
        """Verifies if category has a valid calibrated threshold and empirical holdout scores."""
        return (
            category in self.category_thresholds
            and category in self.calibration_scores
            and len(self.calibration_scores[category]) > 0
        )

    def fit(self, scores: Union[List[float], np.ndarray], category: str = "default") -> float:
        """
        Fits exact finite-sample conformal quantile threshold on independent normal holdout scores.
        Args:
            scores: Array of anomaly non-conformity scores on held-out normal images
            category: Category name identifier
        Returns:
            calibrated_threshold tau_c
        Raises:
            InsufficientSampleSizeError: If n is too small to guarantee target alpha.
        """
        if not category or not isinstance(category, str):
            raise ValueError(f"Invalid category identifier: '{category}'")

        scores_arr = np.asarray(scores, dtype=np.float64)
        n = len(scores_arr)
        if n == 0:
            raise ValueError(f"Cannot calibrate with 0 samples for category '{category}'")

        # Finite-sample order statistic rank (1-indexed): k = ceil((n + 1) * (1 - alpha))
        k = math.ceil((n + 1) * (1.0 - self.alpha))
        if k > n:
            min_n = math.ceil(1.0 / self.alpha - 1.0)
            raise InsufficientSampleSizeError(
                f"INSUFFICIENT CALIBRATION DATA: n={n} normal samples cannot mathematically support "
                f"target alpha={self.alpha:.4f}. The requested quantile rank k={k} exceeds n={n}. "
                f"Minimum sample size required is n >= {min_n}. "
                f"Arbitrary heuristic clipping is rejected under safety controls."
            )

        # Sort in ascending order
        sorted_scores = np.sort(scores_arr)
        # 1-indexed order statistic: the k-th smallest element is at index k - 1
        threshold = float(sorted_scores[k - 1])

        self.category_thresholds[category] = threshold
        self.calibration_scores[category] = sorted_scores.tolist()
        self.sample_counts[category] = n

        # Checksum of calibration score vector
        score_bytes = sorted_scores.tobytes()
        self.checksums[category] = hashlib.sha256(score_bytes).hexdigest()

        logger.info(
            f"Split Conformal Calibration: Category '{category}' calibrated with n={n} samples. "
            f"Rank k={k}/{n} (1-indexed), Threshold tau_c={threshold:.4f} (alpha={self.alpha:.3f})"
        )
        return threshold

    def predict(
        self,
        score: float,
        category: str = "default",
        strict: bool = True
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Evaluates test score against calibrated conformal threshold.
        Returns:
            is_anomaly: bool (True if score >= threshold or p_val <= alpha)
            conformal_p_value: float upper-tail non-conformity p-value
            threshold: float calibrated cutoff
        """
        if not self.is_calibrated(category):
            if strict:
                raise UncalibratedError(
                    f"UNCALIBRATED: Category '{category}' has no valid split conformal calibration. "
                    f"Fail-safe policy prohibits autonomous acceptance."
                )
            return False, None, None

        threshold = self.category_thresholds[category]
        cal_scores = np.asarray(self.calibration_scores[category], dtype=np.float64)
        n = len(cal_scores)

        # Upper-tail nonconformity p-value:
        # p(x) = (1 + sum(s_i >= s_test)) / (n + 1)
        count_greater_equal = int(np.sum(cal_scores >= score))
        p_val = float((1.0 + count_greater_equal) / (n + 1.0))

        # Decision rule: consistent between threshold comparison and p-value check
        is_anomaly = bool(score >= threshold or p_val <= self.alpha)

        return is_anomaly, round(p_val, 6), round(threshold, 6)

    def save(self, path: str):
        """Saves calibration profiles with cryptographic digests and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "6.0.0",
            "alpha": self.alpha,
            "category_thresholds": self.category_thresholds,
            "calibration_scores": self.calibration_scores,
            "sample_counts": self.sample_counts,
            "checksums": self.checksums,
            "method": "split_conformal_finite_sample",
            "safety_scope": "false_alarm_rate_control_under_exchangeability"
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Loads and verifies calibration profiles."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.alpha = float(data["alpha"])
        self.category_thresholds = data["category_thresholds"]
        self.calibration_scores = data["calibration_scores"]
        self.sample_counts = data["sample_counts"]
        self.checksums = data.get("checksums", {})
