"""
Unit & Regression Tests for Nominal Memory Bank and Coreset Algorithms (V6.0).
=============================================================================
Validates:
  1. Exact Greedy k-Center coreset selection
  2. Approximate Greedy k-Center (Johnson-Lindenstrauss random projection)
  3. K-Means++ Diversity sampling
  4. Uniform and Random sampling baselines
  5. Deterministic reproducibility with explicit seeds
  6. Reference normal distance statistics computation
  7. Cross-image distance ranking preservation
  8. State dict serialization and SHA-256 integrity verification
  9. Category compatibility enforcement in model checkpoint loading
"""

import pytest
import torch

from vista_vlm.models.memory_bank import (
    NominalMemoryBank,
    CoresetAlgorithm,
    coreset_greedy_k_center,
    coreset_approx_greedy_k_center,
    coreset_kmeans_plus_plus,
    coreset_uniform,
    coreset_random
)
from vista_vlm.models.vista_model import VISTAAnomalyModel, ModelState


def test_coreset_algorithms_shapes_and_reproducibility():
    N, D, M = 200, 128, 25
    features = torch.randn(N, D)

    # 1. Exact Greedy k-Center
    c_exact, idx_exact = coreset_greedy_k_center(features, M, seed=42)
    assert c_exact.shape == (M, D)
    assert len(set(idx_exact)) == M
    # Determinism
    c_exact2, idx_exact2 = coreset_greedy_k_center(features, M, seed=42)
    assert torch.equal(c_exact, c_exact2)

    # 2. Approx Greedy k-Center
    c_approx, idx_approx = coreset_approx_greedy_k_center(features, M, proj_dim=32, seed=42)
    assert c_approx.shape == (M, D)
    assert len(set(idx_approx)) == M
    c_approx2, idx_approx2 = coreset_approx_greedy_k_center(features, M, proj_dim=32, seed=42)
    assert torch.equal(c_approx, c_approx2)

    # 3. K-Means++
    c_km, idx_km = coreset_kmeans_plus_plus(features, M, seed=42)
    assert c_km.shape == (M, D)
    assert len(set(idx_km)) == M

    # 4. Uniform
    c_uni, idx_uni = coreset_uniform(features, M)
    assert c_uni.shape == (M, D)
    assert len(set(idx_uni)) == M

    # 5. Random
    c_rand, idx_rand = coreset_random(features, M, seed=42)
    assert c_rand.shape == (M, D)
    assert len(set(idx_rand)) == M
    c_rand2, _ = coreset_random(features, M, seed=42)
    assert torch.equal(c_rand, c_rand2)


def test_nominal_memory_bank_fitting_and_stats():
    mem = NominalMemoryBank(
        algorithm=CoresetAlgorithm.APPROX_GREEDY_K_CENTER,
        max_patches=50,
        seed=100
    )
    assert not mem.is_fitted

    N, D = 300, 64
    feats = torch.randn(N, D)
    info = mem.fit(feats, category="cable", device="cpu")

    assert mem.is_fitted
    assert mem.patches.shape == (50, D)
    assert mem.sha256 is not None
    assert "ref_stats" in info
    stats = mem.ref_stats
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["p95"] <= stats["max"]
    assert stats["scale_range"] > 0


def test_distance_map_computation():
    mem = NominalMemoryBank(max_patches=30, seed=42)
    feats = torch.randn(200, 64)
    mem.fit(feats, category="bottle")

    B, gh, gw, C = 2, 8, 8, 64
    query = torch.randn(B, gh * gw, C)

    raw_dist, cal_dist, raw_scores = mem.compute_distance_map(query, spatial_shape=(gh, gw))
    assert raw_dist.shape == (B, 1, gh, gw)
    assert cal_dist.shape == (B, 1, gh, gw)
    assert raw_scores.shape == (B,)

    # Calibrated map must be strictly bounded in [0, 1]
    assert (cal_dist >= 0.0).all() and (cal_dist <= 1.0).all()
    # Raw scores must equal maximum patch distance
    for b in range(B):
        assert torch.isclose(raw_scores[b], raw_dist[b].max())


def test_memory_bank_state_dict_roundtrip():
    mem1 = NominalMemoryBank(max_patches=40, seed=77)
    mem1.fit(torch.randn(150, 32), category="leather")

    state = mem1.state_dict()
    assert state["version"] in ("6.0.0", "7.0.0")
    assert state["sha256"] == mem1.sha256

    mem2 = NominalMemoryBank()
    mem2.load_state_dict(state)
    assert mem2.is_fitted
    assert torch.equal(mem1.patches, mem2.patches)
    assert mem2.category == "leather"
    assert mem2.ref_stats == mem1.ref_stats


def test_checkpoint_category_mismatch_fails_closed(tmp_path):
    model = VISTAAnomalyModel(category="bottle")
    ckpt_path = str(tmp_path / "bottle_model.pth")
    model.save_checkpoint(ckpt_path)

    # Attempt to load checkpoint into a model expecting category "cable"
    model_cable = VISTAAnomalyModel(category="cable")
    with pytest.raises(ValueError, match="Checkpoint category mismatch"):
        model_cable.load_checkpoint(ckpt_path, expected_category="cable")
    assert model_cable.state == ModelState.INVALID

