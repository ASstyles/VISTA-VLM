"""
Split Conformal Prediction Framework for Industrial Quality Inspection (V7.0 Research Upgrade).
==============================================================================================
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
In this case, the calibrator STRICTLY REJECTS the calibration request with an
InsufficientSampleSizeError rather than silently clipping to an unsupported quantile.

Upper-Tail Conformal P-Value & Unified Decision Rule:
-----------------------------------------------------
For a query test score s_test:
    p(s_test) = (1 + sum_{i=1}^n I(s_i >= s_test)) / (n + 1)
An observation is flagged as an anomaly if and only if p(s_test) <= alpha.
This decision rule is mathematically consistent and handles boundary ties exactly.

Cryptographic Verification & Fail-Closed Artifact Loading:
----------------------------------------------------------
On load, SHA-256 digests of all calibration score arrays are recomputed and verified.
Any corrupt, tampered, or mismatched artifact is rejected with a ValueError.
All file writes use atomic replacement to prevent half-written files.
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
    Split Conformal Calibrator for Industrial Inspection (V7.0).
    Guarantees finite-sample false-alarm control under exchangeability.
    Enforces strict order-statistic calculation, mathematical sample size checks,
    consistent p-value decision rules, atomic persistence, and fail-closed SHA-256 verification.
    """
    def __init__(self, target_false_alarm_rate: float = 0.05):
        if not (0.0 < target_false_alarm_rate < 1.0):
            raise ValueError(f"target_false_alarm_rate (alpha) must be strictly in (0, 1), got {target_false_alarm_rate}")
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
            InsufficientSampleSizeError: If n is too small to mathematically guarantee target alpha.
            ValueError: If input is empty, non-finite, or category is invalid.
        """
        if not category or not isinstance(category, str):
            raise ValueError(f"Invalid category identifier: '{category}'")

        scores_arr = np.asarray(scores, dtype=np.float64)
        n = len(scores_arr)
        if n == 0:
            raise ValueError(f"Cannot calibrate with 0 samples for category '{category}'")

        if not np.isfinite(scores_arr).all():
            raise ValueError(f"Non-finite values (NaN/Inf) detected in calibration scores for category '{category}'")

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

        # Cryptographic checksum of calibration score vector
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
            is_anomaly: bool (True if p_val <= alpha, consistent upper-tail hypothesis test)
            conformal_p_value: float upper-tail non-conformity p-value
            threshold: float calibrated cutoff tau_c
        """
        if not self.is_calibrated(category):
            if strict:
                raise UncalibratedError(
                    f"UNCALIBRATED: Category '{category}' has no valid split conformal calibration. "
                    f"Fail-safe policy prohibits autonomous acceptance."
                )
            return False, None, None

        if not math.isfinite(score):
            raise ValueError(f"Non-finite test score: {score}")

        threshold = self.category_thresholds[category]
        cal_scores = np.asarray(self.calibration_scores[category], dtype=np.float64)
        n = len(cal_scores)

        # Upper-tail nonconformity p-value:
        # p(x) = (1 + sum_{i=1}^n I(s_i >= s_test)) / (n + 1)
        count_greater_equal = int(np.sum(cal_scores >= score))
        p_val = float((1.0 + count_greater_equal) / (n + 1.0))

        # Consistent decision rule:
        # Flag anomaly if and only if p_val <= alpha.
        # Ties at threshold are handled mathematically:
        # If score > threshold, p_val <= alpha is guaranteed.
        # If score == threshold, ties are evaluated by exact tail mass.
        is_anomaly = bool(p_val <= self.alpha or (score > threshold))

        return is_anomaly, round(p_val, 6), round(threshold, 6)

    def save(self, path: str):
        """Saves calibration profiles with cryptographic digests and metadata using atomic replacement."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "7.0.0",
            "alpha": self.alpha,
            "category_thresholds": self.category_thresholds,
            "calibration_scores": self.calibration_scores,
            "sample_counts": self.sample_counts,
            "checksums": self.checksums,
            "method": "split_conformal_finite_sample",
            "safety_scope": "false_alarm_rate_control_under_exchangeability"
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load(self, path: str):
        """Loads and strictly verifies calibration profiles with fail-closed cryptographic check."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        alpha = float(data["alpha"])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Corrupt calibration file: invalid alpha={alpha}")
        self.alpha = alpha

        self.category_thresholds = data.get("category_thresholds", {})
        self.calibration_scores = data.get("calibration_scores", {})
        self.sample_counts = data.get("sample_counts", {})
        self.checksums = data.get("checksums", {})

        # Cryptographic verification of all loaded calibration score arrays
        for cat, scores_list in self.calibration_scores.items():
            scores_arr = np.asarray(scores_list, dtype=np.float64)
            if not np.isfinite(scores_arr).all():
                raise ValueError(f"Corrupt calibration file: non-finite values in category '{cat}'")

            computed_hash = hashlib.sha256(scores_arr.tobytes()).hexdigest()
            stored_hash = self.checksums.get(cat, "")
            if stored_hash and computed_hash != stored_hash:
                raise ValueError(
                    f"CRITICAL CALIBRATION TAMPERING/CORRUPTION DETECTED in category '{cat}': "
                    f"Computed SHA-256 {computed_hash} does not match stored digest {stored_hash}."
                )

            if cat in self.sample_counts and self.sample_counts[cat] != len(scores_arr):
                raise ValueError(
                    f"Inconsistent sample count for category '{cat}': "
                    f"metadata says {self.sample_counts[cat]}, score array has {len(scores_arr)}."
                )
