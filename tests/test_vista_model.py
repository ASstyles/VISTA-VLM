import pytest
import torch
from vista_vlm.models.backbones.dinov2 import dinov2_vits14
from vista_vlm.models.fusion.global_local import GlobalLocalFusion
from vista_vlm.models.decoder.segmentation import ProgressiveSegmentationDecoder
from vista_vlm.models.vista_model import VISTAAnomalyModel, DiceLoss

def test_dinov2_backbone_shapes():
    model = dinov2_vits14(out_indices=[4, 8, 12])
    x = torch.randn(2, 3, 518, 518)
    feat_dict = model(x)

    spatial_features = feat_dict["patch_features"]
    cls_tokens = feat_dict["cls_features"]

    # Check 3 intermediate blocks
    assert len(spatial_features) == 3
    assert len(cls_tokens) == 3

    for feat in spatial_features:
        assert feat.shape == (2, 384, 37, 37)

    for cls_tok in cls_tokens:
        assert cls_tok.shape == (2, 384)

def test_global_local_fusion():
    fusion = GlobalLocalFusion(
        in_channels_list=[384, 384, 384],
        proj_dim=256,
        fusion_type="global_local"
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

def test_vista_model_forward():
    model = VISTAAnomalyModel(
        backbone_name="dinov2_vits14",
        embed_dim=384,
        out_indices=[4, 8, 12],
        proj_dim=256,
        image_size=518
    )

    x = torch.randn(2, 3, 518, 518)
    outputs = model(x)

    assert "anomaly_map" in outputs
    assert "image_score" in outputs
    assert outputs["anomaly_map"].shape == (2, 1, 518, 518)
    assert outputs["image_score"].shape == (2,)

    # Values must be bounded in [0, 1] due to sigmoid
    assert (outputs["anomaly_map"] >= 0.0).all() and (outputs["anomaly_map"] <= 1.0).all()
    assert (outputs["image_score"] >= 0.0).all() and (outputs["image_score"] <= 1.0).all()

def test_dice_loss():
    dice = DiceLoss()
    pred = torch.tensor([[[[0.9, 0.9], [0.1, 0.1]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss = dice(pred, target)
    assert loss.item() < 0.2
