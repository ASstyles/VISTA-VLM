import numpy as np
import os
import json
from typing import Dict, Any, List, Tuple, Optional


class UncalibratedOODError(ValueError):
    """Raised when OOD detection is requested for an uncalibrated category without prototypes."""
    pass


class OODDetector:
    """
    Out-of-Distribution (OOD) Detector (Phase 7 & Phase 11).
    Detects whether an inspected image originates from an unseen object distribution,
    severe domain shift, or corrupt acquisition using DINOv2 feature representations.
    Fails closed: Uncalibrated or missing categories trigger OOD alarm, NEVER silent pass.
    """
    def __init__(self, metric: str = "mahalanobis", threshold_percentile: float = 98.0):
        self.metric = metric
        self.threshold_percentile = threshold_percentile
        self.prototypes: Dict[str, np.ndarray] = {}
        self.cov_inv: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}

    def is_fitted(self, category: str = "default") -> bool:
        """Verifies if category has fitted prototype and threshold (and covariance if mahalanobis)."""
        has_proto = category in self.prototypes and category in self.thresholds
        if self.metric == "mahalanobis":
            return has_proto and category in self.cov_inv
        return has_proto

    def fit(self, features: np.ndarray, category: str = "default"):
        """
        Fits normal prototype and covariance on calibration features.
        Args:
            features: [N, D] array of DINOv2 global features
        """
        N, D = features.shape
        if N < 2:
            raise ValueError(f"OOD detector requires at least N>=2 calibration samples to fit covariance (got {N})")

        mu = np.mean(features, axis=0)
        self.prototypes[category] = mu

        if self.metric == "mahalanobis":
            # Shrinkage regularized covariance (Ledoit-Wolf style) guarantees invertibility for any sample size N >= 2
            emp_cov = np.cov(features, rowvar=False) if N > 1 else np.eye(D)
            trace_mean = float(np.trace(emp_cov)) / max(1, D)
            shrinkage = 0.2
            reg_cov = (1.0 - shrinkage) * emp_cov + shrinkage * trace_mean * np.eye(D) + 1e-4 * np.eye(D)
            self.cov_inv[category] = np.linalg.pinv(reg_cov)
            # Compute calibration Mahalanobis distances
            diff = features - mu
            dists = np.sqrt(np.clip(np.sum((diff @ self.cov_inv[category]) * diff, axis=1), 0.0, None))
        else:
            # Cosine distance
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            dists = 1.0 - (feat_norm @ mu_norm)

        threshold = float(np.percentile(dists, self.threshold_percentile))
        self.thresholds[category] = threshold

    def compute_distance(
        self,
        feature: np.ndarray,
        category: str = "default",
        strict: bool = False
    ) -> Tuple[bool, float, float]:
        """
        Computes OOD distance for a single test feature vector [D].
        Fails closed: If category is uncalibrated, treats it as OOD with infinite distance.
        Args:
            feature: [D] feature vector
            category: Target category name
            strict: If True, raises UncalibratedOODError on missing category.
                   If False, returns (True, float('inf'), 0.0) signaling fail-safe OOD state.
        Returns:
            is_ood: bool
            distance: float
            threshold: float
        """
        if not self.is_fitted(category):
            if strict:
                raise UncalibratedOODError(
                    f"UNCALIBRATED_OOD: Category '{category}' has no fitted OOD profile. "
                    f"Fail-safe policy requires human review."
                )
            # Fail closed: treat uncalibrated category as OOD so it is NEVER autonomously accepted
            return True, float("inf"), 0.0

        mu = self.prototypes[category]
        threshold = self.thresholds[category]

        if self.metric == "mahalanobis":
            diff = feature - mu
            dist = float(np.sqrt(np.clip(np.dot(diff @ self.cov_inv[category], diff), 0.0, None)))
        else:
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            f_norm = feature / (np.linalg.norm(feature) + 1e-8)
            dist = float(1.0 - np.dot(f_norm, mu_norm))

        is_ood = bool(dist > threshold)
        return is_ood, round(dist, 4), round(threshold, 4)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "version": "5.0.0",
            "metric": self.metric,
            "threshold_percentile": self.threshold_percentile,
            "thresholds": self.thresholds,
            "prototypes": {cat: arr.tolist() for cat, arr in self.prototypes.items()},
            "cov_inv": {cat: arr.tolist() for cat, arr in self.cov_inv.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOD parameters file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.metric = data["metric"]
        self.threshold_percentile = data["threshold_percentile"]
        self.thresholds = data["thresholds"]
        self.prototypes = {cat: np.array(arr, dtype=np.float32) for cat, arr in data["prototypes"].items()}
        self.cov_inv = {cat: np.array(arr, dtype=np.float32) for cat, arr in data.get("cov_inv", {}).items()}
