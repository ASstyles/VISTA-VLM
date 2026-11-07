import numpy as np
import torch
import os
import json
from typing import Dict, Any, List, Tuple, Optional

class OODDetector:
    """
    Out-of-Distribution (OOD) Detector (Phase 11).
    Detects whether an inspected image originates from an unseen object distribution,
    severe domain shift, or corrupt acquisition using DINOv2 feature representations.
    """
    def __init__(self, metric: str = "mahalanobis", threshold_percentile: float = 98.0):
        self.metric = metric
        self.threshold_percentile = threshold_percentile
        self.prototypes: Dict[str, np.ndarray] = {}
        self.cov_inv: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}

    def fit(self, features: np.ndarray, category: str = "default"):
        """
        Fits normal prototype and covariance on calibration features.
        Args:
            features: [N, D] array of DINOv2 global features
        """
        N, D = features.shape
        mu = np.mean(features, axis=0)
        self.prototypes[category] = mu

        if self.metric == "mahalanobis" and N > D:
            # Regularized covariance
            cov = np.cov(features, rowvar=False) + 1e-4 * np.eye(D)
            self.cov_inv[category] = np.linalg.pinv(cov)
            # Compute calibration distances
            diff = features - mu
            dists = np.sqrt(np.sum((diff @ self.cov_inv[category]) * diff, axis=1))
        else:
            # Cosine distance
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            dists = 1.0 - (feat_norm @ mu_norm)

        threshold = float(np.percentile(dists, self.threshold_percentile))
        self.thresholds[category] = threshold

    def compute_distance(self, feature: np.ndarray, category: str = "default") -> Tuple[bool, float, float]:
        """
        Computes OOD distance for a single test feature vector [D].
        Returns:
            is_ood: bool
            distance: float
            threshold: float
        """
        mu = self.prototypes.get(category)
        if mu is None:
            # If category prototype not fitted, cannot declare OOD
            return False, 0.0, 1.0

        threshold = self.thresholds.get(category, 0.5)

        if self.metric == "mahalanobis" and category in self.cov_inv:
            diff = feature - mu
            dist = float(np.sqrt(np.dot(diff @ self.cov_inv[category], diff)))
        else:
            # Cosine distance
            mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
            f_norm = feature / (np.linalg.norm(feature) + 1e-8)
            dist = float(1.0 - np.dot(f_norm, mu_norm))

        is_ood = bool(dist > threshold)
        return is_ood, round(dist, 4), round(threshold, 4)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert numpy arrays to lists
        data = {
            "metric": self.metric,
            "threshold_percentile": self.threshold_percentile,
            "thresholds": self.thresholds,
            "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOD file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.metric = data["metric"]
        self.threshold_percentile = data["threshold_percentile"]
        self.thresholds = data["thresholds"]
        self.prototypes = {k: np.array(v) for k, v in data["prototypes"].items()}
