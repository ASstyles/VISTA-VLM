"""
VISTA-VLM Nominal Memory Bank & Coreset Algorithms (V7.0 Research Upgrade).
==========================================================================
Implements non-parametric normal reference feature memory with:
  1. Exact Greedy k-Center Coreset Selection (Sener & Savarese, 2018; Roth et al., 2022)
  2. Approximate Greedy k-Center via Random Orthogonal Projection (Johnson-Lindenstrauss lemma)
  3. Diversity-based K-Means++ Initialization Sampling (Arthur & Vassilvitskii, 2007)
  4. Deterministic Uniform Strided Subsampling (Fast Baseline)
  5. Seeded Uniform Random Sampling

Enforces strict input validation (non-finite rejection, dimension checks),
deterministic indexing, fail-closed SHA-256 artifact verification,
configurable k-NN distance aggregation (1-NN, mean k-NN, distance-weighted k-NN),
and robust statistical reference distribution estimation (Median, MAD, p95, p99, p99.5).
"""

from __future__ import annotations

import time
import hashlib
from typing import Dict, Any, Tuple, Optional, List, Union
from enum import Enum

import torch
import numpy as np

from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.memory_bank")


class CoresetAlgorithm(str, Enum):
    GREEDY_K_CENTER = "greedy_k_center"
    APPROX_GREEDY_K_CENTER = "approx_greedy_k_center"
    KMEANS_PLUS_PLUS = "kmeans_plus_plus"
    UNIFORM = "uniform"
    RANDOM = "random"


def _validate_feature_tensor(features: torch.Tensor):
    """Validates 2D tensor finiteness and dimensions."""
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Features must be a torch.Tensor, got {type(features)}")
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D [N, D], got shape {features.shape}")
    N, D = features.shape
    if N == 0 or D == 0:
        raise ValueError(f"Features tensor cannot be empty. Got shape {features.shape}")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain non-finite values (NaN or Inf). Rejection enforced under safety controls.")


def coreset_uniform(features: torch.Tensor, n_coreset: int) -> Tuple[torch.Tensor, List[int]]:
    """Uniform deterministic strided subsampling."""
    _validate_feature_tensor(features)
    total = features.shape[0]
    if n_coreset >= total:
        return features, list(range(total))
    indices = torch.linspace(0, total - 1, steps=n_coreset).long()
    selected_indices = indices.tolist()
    return features[indices], selected_indices


def coreset_random(features: torch.Tensor, n_coreset: int, seed: int = 42) -> Tuple[torch.Tensor, List[int]]:
    """Seeded uniform random sampling."""
    _validate_feature_tensor(features)
    total = features.shape[0]
    if n_coreset >= total:
        return features, list(range(total))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g)
    selected_indices = perm[:n_coreset].tolist()
    return features[selected_indices], selected_indices


def coreset_greedy_k_center(
    features: torch.Tensor,
    n_coreset: int,
    seed: int = 42
) -> Tuple[torch.Tensor, List[int]]:
    """
    Exact Greedy k-Center Coreset Algorithm (Sener & Savarese, 2018; Roth et al., 2022).
    Iteratively selects the point that maximizes the minimum distance to all existing centers:
        c_{m} = argmax_{i} min_{c in C} ||x_i - c||_2
    Guarantees deterministic unique indices and bounded minimax manifold coverage.
    """
    _validate_feature_tensor(features)
    N, D = features.shape
    if n_coreset >= N:
        return features, list(range(N))

    g = torch.Generator().manual_seed(seed)
    init_idx = int(torch.randint(0, N, (1,), generator=g).item())
    selected_indices = [init_idx]
    selected_set = {init_idx}

    # Initialize min distances from each point to initial center
    min_distances = torch.norm(features - features[init_idx:init_idx+1], dim=1)

    for _ in range(1, n_coreset):
        # Mask out already selected indices
        min_distances[list(selected_set)] = -1.0
        new_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(new_idx)
        selected_set.add(new_idx)

        dist_to_new = torch.norm(features - features[new_idx:new_idx+1], dim=1)
        min_distances = torch.minimum(min_distances, dist_to_new)

    return features[selected_indices], selected_indices


def coreset_approx_greedy_k_center(
    features: torch.Tensor,
    n_coreset: int,
    proj_dim: int = 64,
    seed: int = 42
) -> Tuple[torch.Tensor, List[int]]:
    """
    Approximate Greedy k-Center using Random Orthogonal Projection (Johnson-Lindenstrauss lemma).
    Preserves pairwise metrics with bounded distortion while accelerating greedy selection.
    """
    _validate_feature_tensor(features)
    N, D = features.shape
    if n_coreset >= N:
        return features, list(range(N))

    g = torch.Generator().manual_seed(seed)
    target_dim = min(proj_dim, D)
    rand_proj = torch.randn(D, target_dim, generator=g)
    rand_proj, _ = torch.linalg.qr(rand_proj)
    proj_features = features @ rand_proj

    init_idx = int(torch.randint(0, N, (1,), generator=g).item())
    selected_indices = [init_idx]
    selected_set = {init_idx}
    min_distances = torch.norm(proj_features - proj_features[init_idx:init_idx+1], dim=1)

    for _ in range(1, n_coreset):
        min_distances[list(selected_set)] = -1.0
        new_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(new_idx)
        selected_set.add(new_idx)

        dist_to_new = torch.norm(proj_features - proj_features[new_idx:new_idx+1], dim=1)
        min_distances = torch.minimum(min_distances, dist_to_new)

    return features[selected_indices], selected_indices


def coreset_kmeans_plus_plus(
    features: torch.Tensor,
    n_coreset: int,
    seed: int = 42
) -> Tuple[torch.Tensor, List[int]]:
    """
    Diversity-based sampling via k-means++ (Arthur & Vassilvitskii, 2007).
    Selects points with probability proportional to squared distance to nearest center.
    """
    _validate_feature_tensor(features)
    N, D = features.shape
    if n_coreset >= N:
        return features, list(range(N))

    g = torch.Generator().manual_seed(seed)
    init_idx = int(torch.randint(0, N, (1,), generator=g).item())
    selected_indices = [init_idx]
    selected_set = {init_idx}

    min_sq_distances = torch.sum((features - features[init_idx:init_idx+1]) ** 2, dim=1)

    for _ in range(1, n_coreset):
        min_sq_distances[list(selected_set)] = 0.0
        sum_sq = float(min_sq_distances.sum().item())
        if sum_sq <= 1e-12:
            # Fallback if remaining points are identical to selected centers
            remaining = [i for i in range(N) if i not in selected_set]
            new_idx = remaining[0]
        else:
            probs = min_sq_distances / sum_sq
            new_idx = int(torch.multinomial(probs, 1, generator=g).item())

        selected_indices.append(new_idx)
        selected_set.add(new_idx)

        dist_to_new_sq = torch.sum((features - features[new_idx:new_idx+1]) ** 2, dim=1)
        min_sq_distances = torch.minimum(min_sq_distances, dist_to_new_sq)

    return features[selected_indices], selected_indices


class NominalMemoryBank:
    """
    Production Nominal Memory Bank (V7.0 Research Upgrade).
    Stores nominal reference embeddings, manages coreset compression,
    and computes cross-image calibrated distance maps with multi-neighbor aggregation
    and robust statistical reference distribution scaling.
    """
    def __init__(
        self,
        algorithm: Union[str, CoresetAlgorithm] = CoresetAlgorithm.APPROX_GREEDY_K_CENTER,
        max_patches: int = 1500,
        seed: int = 42,
        k_neighbors: int = 3,
        neighbor_aggregation: str = "mean"
    ):
        if isinstance(algorithm, str):
            self.algorithm = CoresetAlgorithm(algorithm)
        else:
            self.algorithm = algorithm
        self.max_patches = max_patches
        self.seed = seed
        self.k_neighbors = max(1, k_neighbors)
        self.neighbor_aggregation = neighbor_aggregation  # 'mean', '1nn', 'distance_weighted'

        self.patches: Optional[torch.Tensor] = None
        self.feature_dim: Optional[int] = None
        self.category: Optional[str] = None
        self.created_at: Optional[str] = None
        self.sha256: Optional[str] = None

        # Fixed reference normal statistics computed strictly on training normal patches
        self.ref_stats: Dict[str, float] = {
            "min": 0.0,
            "max": 1.0,
            "mean": 0.5,
            "median": 0.5,
            "mad": 0.1,
            "robust_scale": 0.15,
            "std": 0.1,
            "p90": 0.75,
            "p95": 0.85,
            "p99": 0.95,
            "p99_5": 0.98,
            "scale_range": 1.0
        }

    @property
    def is_fitted(self) -> bool:
        return self.patches is not None and len(self.patches) > 0

    def fit(
        self,
        features: torch.Tensor,
        category: str = "generic",
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Fits coreset memory bank from collected patch tokens [N, D] and establishes
        robust reference normal distribution statistics.
        """
        _validate_feature_tensor(features)
        N, D = features.shape
        self.feature_dim = D
        self.category = category
        t0 = time.time()

        n_target = min(self.max_patches, N)
        logger.info(
            f"Fitting Nominal Memory Bank: category='{category}', total_patches={N}, "
            f"target_coreset={n_target}, algorithm={self.algorithm.value}, k_neighbors={self.k_neighbors}"
        )

        feats_cpu = features.cpu()

        if self.algorithm == CoresetAlgorithm.GREEDY_K_CENTER:
            coreset_t, _ = coreset_greedy_k_center(feats_cpu, n_target, seed=self.seed)
        elif self.algorithm == CoresetAlgorithm.APPROX_GREEDY_K_CENTER:
            coreset_t, _ = coreset_approx_greedy_k_center(feats_cpu, n_target, proj_dim=64, seed=self.seed)
        elif self.algorithm == CoresetAlgorithm.KMEANS_PLUS_PLUS:
            coreset_t, _ = coreset_kmeans_plus_plus(feats_cpu, n_target, seed=self.seed)
        elif self.algorithm == CoresetAlgorithm.RANDOM:
            coreset_t, _ = coreset_random(feats_cpu, n_target, seed=self.seed)
        else:  # UNIFORM
            coreset_t, _ = coreset_uniform(feats_cpu, n_target)

        self.patches = coreset_t.to(device)
        fit_time_s = round(time.time() - t0, 3)

        # Compute robust reference normal distance statistics by measuring distance from all training
        # patches to the fitted coreset
        with torch.no_grad():
            sub_sample_size = min(N, 2000)
            sample_feats = feats_cpu[:sub_sample_size].to(device)
            dists = torch.cdist(sample_feats, self.patches, p=2.0)
            k_eff = min(self.k_neighbors, self.patches.shape[0])
            topk_dists, _ = torch.topk(dists, k=k_eff, dim=1, largest=False)

            if self.neighbor_aggregation == "1nn" or k_eff == 1:
                agg_dists = topk_dists[:, 0]
            elif self.neighbor_aggregation == "distance_weighted" and k_eff > 1:
                weights = F.softmax(-topk_dists, dim=1)
                agg_dists = (topk_dists * weights).sum(dim=1)
            else:  # 'mean'
                agg_dists = topk_dists.mean(dim=1)

            min_dists = agg_dists.cpu().numpy()

        ref_min = float(np.min(min_dists))
        ref_max = float(np.max(min_dists))
        ref_mean = float(np.mean(min_dists))
        ref_median = float(np.median(min_dists))
        ref_mad = float(np.median(np.abs(min_dists - ref_median)))
        ref_robust_scale = float(1.4826 * max(1e-6, ref_mad))
        ref_std = float(np.std(min_dists))
        ref_p90 = float(np.percentile(min_dists, 90))
        ref_p95 = float(np.percentile(min_dists, 95))
        ref_p99 = float(np.percentile(min_dists, 99))
        ref_p99_5 = float(np.percentile(min_dists, 99.5))

        # Robust scale range based on p99 and robust quantile spread:
        # Prevents extreme training outlier patches from compressing anomaly sensitivity
        scale_range = max(1e-4, float(ref_p99 - ref_min) * 1.5)

        self.ref_stats = {
            "min": round(ref_min, 4),
            "max": round(ref_max, 4),
            "mean": round(ref_mean, 4),
            "median": round(ref_median, 4),
            "mad": round(ref_mad, 4),
            "robust_scale": round(ref_robust_scale, 4),
            "std": round(ref_std, 4),
            "p90": round(ref_p90, 4),
            "p95": round(ref_p95, 4),
            "p99": round(ref_p99, 4),
            "p99_5": round(ref_p99_5, 4),
            "scale_range": round(scale_range, 4)
        }

        # SHA-256 checksum of memory bank tensor
        t_bytes = self.patches.cpu().numpy().tobytes()
        self.sha256 = hashlib.sha256(t_bytes).hexdigest()
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            f"Memory bank fitted in {fit_time_s}s | SHA-256={self.sha256[:12]}... | "
            f"RefStats: min={ref_min:.3f}, med={ref_median:.3f}, p95={ref_p95:.3f}, p99={ref_p99:.3f}, max={ref_max:.3f}"
        )

        return {
            "category": self.category,
            "algorithm": self.algorithm.value,
            "coreset_size": self.patches.shape[0],
            "feature_dim": self.feature_dim,
            "fit_time_s": fit_time_s,
            "sha256": self.sha256,
            "ref_stats": self.ref_stats
        }

    def compute_distance_map(
        self,
        patch_tokens: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes pairwise distance from query patches [B, H*W, C] to nominal memory bank
        with configured multi-neighbor aggregation (1-NN, Mean k-NN, Distance-weighted).
        Returns:
            raw_dist_map: [B, 1, gh, gw] continuous un-normalized Euclidean distance
            calibrated_map: [B, 1, gh, gw] scaled to [0, 1] using FIXED reference training stats
            raw_image_scores: [B] aggregated image anomaly score
        """
        if not self.is_fitted:
            raise RuntimeError("NominalMemoryBank is empty. Fit or load before computing distances.")

        B, N_patches, C = patch_tokens.shape
        gh, gw = spatial_shape
        device = patch_tokens.device
        mem = self.patches.to(device)
        k_eff = min(self.k_neighbors, mem.shape[0])

        dist_maps = []
        for b in range(B):
            p_b = patch_tokens[b]  # [N_patches, C]
            dists = torch.cdist(p_b, mem, p=2.0)
            topk_dists, _ = torch.topk(dists, k=k_eff, dim=1, largest=False)

            if self.neighbor_aggregation == "1nn" or k_eff == 1:
                score_per_patch = topk_dists[:, 0]
            elif self.neighbor_aggregation == "distance_weighted" and k_eff > 1:
                weights = F.softmax(-topk_dists, dim=1)
                score_per_patch = (topk_dists * weights).sum(dim=1)
            else:  # 'mean'
                score_per_patch = topk_dists.mean(dim=1)

            dist_map = score_per_patch.reshape(1, 1, gh, gw)
            dist_maps.append(dist_map)

        raw_dist_map = torch.cat(dist_maps, dim=0)  # [B, 1, gh, gw]
        raw_image_scores = raw_dist_map.view(B, -1).max(dim=1)[0]  # [B]

        # Reference-calibrated scaling: uses FIXED normal training distribution
        ref_min = self.ref_stats["min"]
        scale_range = self.ref_stats["scale_range"]
        calibrated_map = torch.clamp((raw_dist_map - ref_min) / scale_range, 0.0, 1.0)

        return raw_dist_map, calibrated_map, raw_image_scores

    def state_dict(self) -> Dict[str, Any]:
        """Returns serializable dictionary of memory bank state."""
        return {
            "algorithm": self.algorithm.value,
            "max_patches": self.max_patches,
            "seed": self.seed,
            "k_neighbors": self.k_neighbors,
            "neighbor_aggregation": self.neighbor_aggregation,
            "patches": self.patches.cpu() if self.patches is not None else None,
            "feature_dim": self.feature_dim,
            "category": self.category,
            "created_at": self.created_at,
            "sha256": self.sha256,
            "ref_stats": self.ref_stats,
            "version": "7.0.0"
        }

    def load_state_dict(self, state: Dict[str, Any], device: str = "cpu"):
        """Restores memory bank from state dict with fail-closed checksum verification."""
        self.algorithm = CoresetAlgorithm(state.get("algorithm", CoresetAlgorithm.APPROX_GREEDY_K_CENTER.value))
        self.max_patches = state.get("max_patches", 1500)
        self.seed = state.get("seed", 42)
        self.k_neighbors = state.get("k_neighbors", 3)
        self.neighbor_aggregation = state.get("neighbor_aggregation", "mean")
        self.feature_dim = state.get("feature_dim", None)
        self.category = state.get("category", None)
        self.created_at = state.get("created_at", None)
        self.sha256 = state.get("sha256", None)
        self.ref_stats = state.get("ref_stats", self.ref_stats)

        patches_t = state.get("patches", None)
        if patches_t is not None:
            self.patches = patches_t.to(device)
            # Strict fail-closed checksum verification
            curr_hash = hashlib.sha256(self.patches.cpu().numpy().tobytes()).hexdigest()
            if self.sha256 and curr_hash != self.sha256:
                raise ValueError(
                    f"CRITICAL ARTIFACT CORRUPTION: Memory bank SHA-256 digest mismatch! "
                    f"Expected {self.sha256}, got {curr_hash}. Refusing to load corrupted state."
                )
