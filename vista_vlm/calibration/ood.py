"""
Out-of-Distribution (OOD) Detector (V6.0).
==========================================
Identifies whether an inspected image originates from an unseen object distribution,
severe optical/camera misalignment, foreign contaminant, or novel product class
using DINOv2 foundation representations.

Features:
  1. Regularized Ledoit-Wolf Shrinkage Covariance for strictly positive-definite metric tensors
  2. SVD / Condition Number Health Auditing (detecting singular/degenerate manifolds)
  3. Strict metric separation (never compares Mahalanobis distance with cosine threshold)
  4. Fails Closed: Uncalibrated or missing categories trigger OOD alarm, NEVER silent pass
  5. Cryptographic checksums and version metadata
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Tuple
import numpy as np

from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.ood")


class UncalibratedOODError(ValueError):
    """Raised when OOD detection is evaluated for an uncalibrated category."""
    pass


class InvalidFeatureError(ValueError):
    """Raised when query features contain non-finite values or dimension mismatch."""
    pass


class OODDetector:
    """
    Production Out-of-Distribution Detector (V6.0).
    Enforces regularized covariance invertibility and strict fail-closed safety semantics.
    """
    def __init__(
        self,
        metric: str = "mahalanobis",
        threshold_percentile: float = 98.0,
        shrinkage: float = 0.25
    ):
        if metric not in ("mahalanobis", "cosine"):
            raise ValueError(f"Unsupported OOD metric: '{metric}'. Must be 'mahalanobis' or 'cosine'")
        self.metric = metric
        self.threshold_percentile = float(threshold_percentile)
        self.shrinkage = float(shrinkage)

        self.prototypes: Dict[str, np.ndarray] = {}
        self.cov_inv: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}
        self.feature_dims: Dict[str, int] = {}
        self.checksums: Dict[str, str] = {}

    def is_fitted(self, category: str = "default") -> bool:
        """Verifies if category has fitted prototype and threshold."""
        has_proto = category in self.prototypes and category in self.thresholds
        if self.metric == "mahalanobis":
            return has_proto and category in self.cov_inv
        return has_proto

    def fit(self, features: np.ndarray, category: str = "default"):
        """
        Fits normal prototype and regularized covariance on calibration feature vectors [N, D].
        """
        features_arr = np.asarray(features, dtype=np.float64)
        if not np.all(np.isfinite(features_arr)):
            raise InvalidFeatureError(f"Calibration features for category '{category}' contain non-finite values (NaN/Inf)")

        N, D = features_arr.shape
        if N < 2:
            raise ValueError(f"OOD detector requires at least N>=2 calibration vectors, got N={N}")

        self.feature_dims[category] = D
        mu = np.mean(features_arr, axis=0)
        self.prototypes[category] = mu

        if self.metric == "mahalanobis":
            # Empirical covariance
            emp_cov = np.cov(features_arr, rowvar=False) if N > 1 else np.eye(D)
            trace_mean = float(np.trace(emp_cov)) / max(1, D)
            # Ledoit-Wolf style shrinkage towards identity target
            reg_cov = (1.0 - self.shrinkage) * emp_cov + self.shrinkage * trace_mean * np.eye(D) + 1e-4 * np.eye(D)

            # Robust pseudo-inverse with condition number validation
            cond_num = np.linalg.cond(reg_cov)
            if cond_num > 1e12:
                logger.warning(f"OOD Covariance matrix for '{category}' is ill-conditioned (cond={cond_num:.2e}). Applying SVD clipping.")
                u, s, vh = np.linalg.svd(reg_cov)
                s_clamped = np.maximum(s, 1e-3)
                inv_cov = (vh.T / s_clamped) @ u.T
            else:
                inv_cov = np.linalg.pinv(reg_cov)

            self.cov_inv[category] = inv_cov

            # Compute calibration Mahalanobis distances
            diff = features_arr - mu
            dists = np.sqrt(np.clip(np.sum((diff @ inv_cov) * diff, axis=1), 0.0, None))
        else:
            # Cosine distance
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            feat_norm = features_arr / (np.linalg.norm(features_arr, axis=1, keepdims=True) + 1e-8)
            dists = 1.0 - (feat_norm @ mu_norm)

        threshold = float(np.percentile(dists, self.threshold_percentile))
        self.thresholds[category] = threshold

        # Checksum
        bytes_sig = mu.tobytes() + np.array([threshold]).tobytes()
        self.checksums[category] = hashlib.sha256(bytes_sig).hexdigest()

        logger.info(
            f"OOD Profile fitted: category='{category}', N={N}, D={D}, metric={self.metric}, "
            f"threshold={threshold:.4f} ({self.threshold_percentile}th percentile)"
        )

    def compute_distance(
        self,
        feature: np.ndarray,
        category: str = "default",
        strict: bool = False
    ) -> Tuple[bool, float, float]:
        """
        Computes OOD distance for query feature [D].
        Fails closed: Uncalibrated categories or invalid features return (True, inf, threshold).
        """
        if not self.is_fitted(category):
            if strict:
                raise UncalibratedOODError(
                    f"UNCALIBRATED_OOD: Category '{category}' has no fitted OOD profile. "
                    f"Fail-safe policy prohibits autonomous acceptance."
                )
            return True, float("inf"), 0.0

        feat_arr = np.asarray(feature, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(feat_arr)):
            logger.error(f"OOD evaluation received non-finite feature vector for category '{category}'")
            return True, float("inf"), self.thresholds[category]

        expected_d = self.feature_dims.get(category, len(feat_arr))
        if len(feat_arr) != expected_d:
            raise InvalidFeatureError(
                f"Feature dimension mismatch for category '{category}': expected {expected_d}, got {len(feat_arr)}"
            )

        mu = self.prototypes[category]
        threshold = self.thresholds[category]

        if self.metric == "mahalanobis":
            diff = feat_arr - mu
            dist_sq = float(np.dot(diff @ self.cov_inv[category], diff))
            dist = float(np.sqrt(max(0.0, dist_sq)))
        else:
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            f_norm = feat_arr / (np.linalg.norm(feat_arr) + 1e-8)
            dist = float(1.0 - np.dot(f_norm, mu_norm))

        is_ood = bool(dist > threshold)
        return is_ood, round(dist, 4), round(threshold, 4)

    def save(self, path: str):
        """Saves OOD profiles to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "6.0.0",
            "metric": self.metric,
            "threshold_percentile": self.threshold_percentile,
            "shrinkage": self.shrinkage,
            "thresholds": self.thresholds,
            "feature_dims": self.feature_dims,
            "checksums": self.checksums,
            "prototypes": {cat: arr.tolist() for cat, arr in self.prototypes.items()},
            "cov_inv": {cat: arr.tolist() for cat, arr in self.cov_inv.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Loads OOD profiles from JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOD parameters file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.metric = data["metric"]
        self.threshold_percentile = float(data["threshold_percentile"])
        self.shrinkage = float(data.get("shrinkage", 0.25))
        self.thresholds = data["thresholds"]
        self.feature_dims = data.get("feature_dims", {})
        self.checksums = data.get("checksums", {})
        self.prototypes = {cat: np.array(arr, dtype=np.float64) for cat, arr in data["prototypes"].items()}
        self.cov_inv = {cat: np.array(arr, dtype=np.float64) for cat, arr in data.get("cov_inv", {}).items()}
