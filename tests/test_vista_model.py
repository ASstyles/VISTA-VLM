import pytest
import os
import tempfile
import torch
from vista_vlm.models.backbones.dinov2 import dinov2_vits14
from vista_vlm.models.fusion.global_local import GlobalLocalFusion
from vista_vlm.models.decoder.segmentation import ProgressiveSegmentationDecoder
from vista_vlm.models.vista_model import (
    VISTAAnomalyModel,
    compute_image_anomaly_score,
    DiceLoss,
    TverskyLoss,
    ModelState
)

def test_dinov2_backbone_shapes():
    model = dinov2_vits14(out_indices=[4, 8, 12])
    x = torch.randn(2, 3, 518, 518)
    feat_dict = model(x)

    spatial_features = feat_dict["patch_features"]
    cls_tokens = feat_dict["cls_features"]

    assert len(spatial_features) == 3
    assert len(cls_tokens) == 3

    for feat in spatial_features:
        assert feat.shape == (2, 384, 37, 37)

    for cls_tok in cls_tokens:
        assert cls_tok.shape == (2, 384)

@pytest.mark.parametrize("fusion_type", ["film", "gated", "cross_attention", "concat_proj", "local_only"])
def test_global_local_fusion_variants(fusion_type):
    fusion = GlobalLocalFusion(
        in_channels_list=[384, 384, 384],
        proj_dim=256,
        fusion_type=fusion_type
    )

    feat_maps = [
        torch.randn(2, 384, 37, 37),
        torch.randn(2, 384, 37, 37),
        torch.randn(2, 384, 37, 37)
    ]
    cls_tokens = [torch.randn(2, 384), torch.randn(2, 384), torch.randn(2, 384)]

    out = fusion(feat_maps, cls_tokens)
    assert out.shape == (2, 256, 37, 37)

def test_segmentation_decoder():
    decoder = ProgressiveSegmentationDecoder(
        in_channels=256,
        target_size=518
    )

    feat = torch.randn(2, 256, 37, 37)
    mask = decoder(feat, out_size=(518, 518))
    assert mask.shape == (2, 1, 518, 518)

def test_segmentation_decoder_with_guide_map():
    decoder = ProgressiveSegmentationDecoder(
        in_channels=256,
        target_size=518,
        guide_in_channels=1
    )
    feat = torch.randn(2, 256, 37, 37)
    guide = torch.rand(2, 1, 37, 37)
    mask = decoder(feat, guide_map=guide, out_size=(518, 518))
    assert mask.shape == (2, 1, 518, 518)

@pytest.mark.parametrize("scoring_mode", ["hybrid", "top_k", "max", "percentile_99", "memory_max", "memory_topk"])
def test_vista_model_forward_scoring_modes(scoring_mode):
    model = VISTAAnomalyModel(
        backbone_name="dinov2_vits14",
        embed_dim=384,
        out_indices=[4, 8, 12],
        proj_dim=256,
        image_size=518,
        scoring_mode=scoring_mode
    )

    x = torch.randn(2, 3, 518, 518)
    outputs = model(x)

    assert "anomaly_map" in outputs
    assert "image_score" in outputs
    assert outputs["anomaly_map"].shape == (2, 1, 518, 518)
    assert outputs["image_score"].shape == (2,)

    assert (outputs["anomaly_map"] >= 0.0).all() and (outputs["anomaly_map"] <= 1.0).all()
    assert (outputs["image_score"] >= 0.0).all() and (outputs["image_score"] <= 1.0).all()

def test_dice_loss():
    dice = DiceLoss()
    pred = torch.tensor([[[[0.9, 0.9], [0.1, 0.1]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss = dice(pred, target)
    assert loss.item() < 0.2

def test_tversky_loss():
    tversky = TverskyLoss(alpha=0.7, beta=0.3)
    pred = torch.tensor([[[[0.9, 0.9], [0.1, 0.1]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss = tversky(pred, target)
    assert loss.item() < 0.2
    assert 0.0 <= loss.item() <= 1.0

def test_model_state_lifecycle():
    model = VISTAAnomalyModel()
    assert model.state == ModelState.READY
    assert model.is_ready is True


def test_compute_image_anomaly_score_bounded_and_monotonic():
    B, H, W = 4, 32, 32
    spatial_map = torch.rand(B, 1, H, W)
    head_score = torch.rand(B)
    raw_dist = torch.tensor([1.0, 2.0, 3.0, 5.0])
    ref_stats = {"min": 1.0, "scale_range": 2.0}

    score, components = compute_image_anomaly_score(
        anomaly_map=spatial_map,
        head_score=head_score,
        raw_memory_distance=raw_dist,
        memory_ref_stats=ref_stats,
        scoring_mode="hybrid"
    )

    assert score.shape == (B,)
    assert (score >= 0.0).all() and (score <= 1.0).all()
    assert "calibrated_memory_score" in components
    assert "spatial_score" in components
    assert "head_score" in components

    # Monotonicity check on calibrated memory component
    cal_mem = components["calibrated_memory_score"]
    assert cal_mem[0] <= cal_mem[1] <= cal_mem[2] <= cal_mem[3]
    # Check that d=min gives 0.0
    assert torch.isclose(cal_mem[0], torch.tensor(0.0), atol=1e-5)


@pytest.mark.parametrize("mode", [
    "hybrid", "head_only", "memory_only", "spatial_only",
    "global_spatial", "memory_spatial", "max", "percentile_99"
])
def test_compute_image_anomaly_score_modes(mode):
    B = 3
    spatial_map = torch.rand(B, 1, 16, 16)
    head_score = torch.rand(B)
    raw_dist = torch.rand(B) * 2.0
    ref_stats = {"min": 0.5, "scale_range": 1.5}

    score, comp = compute_image_anomaly_score(
        anomaly_map=spatial_map,
        head_score=head_score,
        raw_memory_distance=raw_dist,
        memory_ref_stats=ref_stats,
        scoring_mode=mode
    )
    assert score.shape == (B,)
    assert (score >= 0.0).all() and (score <= 1.0).all()


def test_forward_and_compute_image_score_identity():
    """Verify that forward() and compute_image_score() yield identical scores."""
    model = VISTAAnomalyModel(image_size=126, top_k_ratio=0.05)
    model.eval()
    x = torch.randn(2, 3, 126, 126)

    with torch.no_grad():
        out = model(x)
        forward_score = out["image_score"]

        # Call compute_image_score directly using outputs from model
        manual_score, _ = model.compute_image_score(
            anomaly_map=out["anomaly_map"],
            head_score=out["score_components"]["head_score"],
            raw_memory_distance=out["score_components"].get("raw_memory_distance")
        )

    assert torch.allclose(forward_score, manual_score, atol=1e-6)


def test_checkpoint_roundtrip_preserves_scoring_behavior():
    """Verify that saving and reloading a checkpoint strictly preserves scoring behavior."""
    model = VISTAAnomalyModel(
        image_size=126,
        memory_weight=0.45,
        spatial_weight=0.25,
        global_head_weight=0.30,
        scoring_mode="hybrid"
    )
    model.eval()

    # Artificially fit dummy memory bank
    dummy_patches = torch.randn(50, 768)
    model.nominal_memory.patches = dummy_patches
    model.nominal_memory.ref_stats = {"min": 1.2, "scale_range": 2.5}
    assert model.is_memory_guided is True

    x = torch.randn(2, 3, 126, 126)
    with torch.no_grad():
        orig_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_test.pt")
        model.save_checkpoint(ckpt_path)

        # Reload into fresh model instance
        reloaded_model = VISTAAnomalyModel(image_size=126)
        reloaded_model.load_checkpoint(ckpt_path)
        reloaded_model.eval()

        assert reloaded_model.is_memory_guided is True
        assert reloaded_model.memory_weight == 0.45
        assert reloaded_model.spatial_weight == 0.25
        assert reloaded_model.global_head_weight == 0.30
        assert reloaded_model.scoring_mode == "hybrid"

        with torch.no_grad():
            reloaded_out = reloaded_model(x)

    assert torch.allclose(orig_out["image_score"], reloaded_out["image_score"], atol=1e-5)
    assert torch.allclose(orig_out["anomaly_map"], reloaded_out["anomaly_map"], atol=1e-5)
