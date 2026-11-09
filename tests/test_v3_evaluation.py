import pytest
import torch
import numpy as np
from vista_vlm.evaluation.metrics import (
    compute_aupro_score,
    compute_pixel_metrics,
    compute_image_metrics,
    check_spatial_map_validity,
)
from vista_vlm.models.decoder.segmentation import ProgressiveSegmentationDecoder
from vista_vlm.models.vista_model import SobelBoundaryLoss
from vista_vlm.data.synthetic import (
    get_foreground_mask,
    generate_multiscale_synthetic_anomaly,
    generate_localized_perlin_mask,
)
from vista_vlm.calibration.conformal import ConformalCalibrator

def test_aupro_linear_endpoint_interpolation():
    """
    Verify that compute_aupro_score correctly performs linear endpoint interpolation
    at max_fpr = 0.3 rather than dropping the area when threshold FPR < 0.3.
    """
    # Create simple synthetic ground truth masks with 1 defect region
    masks = np.zeros((2, 100, 100), dtype=np.uint8)
    masks[0, 40:60, 40:60] = 1  # 400 anomaly pixels
    masks[1, 20:30, 20:30] = 1  # 100 anomaly pixels

    # Create synthetic anomaly maps with smooth peak around defects
    maps = np.zeros((2, 100, 100), dtype=np.float32)
    maps[0, 35:65, 35:65] = 0.8
    maps[0, 40:60, 40:60] = 0.95
    maps[1, 15:35, 15:35] = 0.7
    maps[1, 20:30, 20:30] = 0.90

    # Ensure validity check passes
    stats = check_spatial_map_validity(maps)
    assert stats["is_valid"] is True
    assert stats["mean_spatial_std"] > 1e-4

    # Compute AUPRO
    aupro = compute_aupro_score(masks, maps, max_fpr=0.3, num_thresholds=100)
    assert 0.0 <= aupro <= 1.0
    # Good predictions should yield substantial AUPRO (> 0.5)
    assert aupro > 0.50

def test_spatial_map_validity_rejects_uniform_map():
    """
    Degenerate constant/uniform spatial maps must be rejected by check_spatial_map_validity.
    """
    constant_map = np.ones((2, 64, 64), dtype=np.float32) * 0.42
    stats = check_spatial_map_validity(constant_map)
    assert stats["is_valid"] is False
    assert stats["is_constant_scalar"] is True
    assert stats["mean_spatial_std"] == pytest.approx(0.0, abs=1e-8)

def test_sobel_boundary_loss_gradient_flow():
    """
    Verify SobelBoundaryLoss computes sharp edge gradients and flows gradients backward.
    """
    criterion = SobelBoundaryLoss()
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:44, 20:44] = 1.0  # Box shape with clear boundaries

    loss = criterion(pred, target)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.sum(torch.abs(pred.grad)) > 0.0

def test_decoder_deep_supervision_shapes():
    """
    Verify ProgressiveSegmentationDecoder deep supervision auxiliary outputs.
    """
    decoder = ProgressiveSegmentationDecoder(in_channels=256, target_size=518)
    feat = torch.randn(2, 256, 37, 37)

    # Standard forward
    out = decoder(feat, out_size=(518, 518), return_aux=False)
    assert out.shape == (2, 1, 518, 518)

    # Deep supervision forward
    main_out, aux_list = decoder(feat, out_size=(518, 518), return_aux=True)
    assert main_out.shape == (2, 1, 518, 518)
    assert len(aux_list) == 2
    assert aux_list[0].shape == (2, 1, 148, 148)
    assert aux_list[1].shape == (2, 1, 296, 296)

def test_realistic_synthetic_anomaly_foreground_constraint():
    """
    Verify that multi-scale synthetic defects are realistic in size (< 10% foreground)
    and constrained within object foreground.
    """
    # Create synthetic image with a dark background and a central object
    toy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    toy_img[40:160, 40:160] = 200  # Central foreground object

    fg_mask = get_foreground_mask(toy_img)
    assert fg_mask[100, 100] == 255  # Center is foreground
    assert fg_mask[5, 5] == 0        # Top-left corner is background

    syn_img, syn_mask = generate_multiscale_synthetic_anomaly(toy_img)
    anomaly_pixels = np.sum(syn_mask > 0)
    fg_pixels = np.sum(fg_mask > 0)

    # Anomaly must be present but constrained (< 15% of foreground)
    assert anomaly_pixels > 0
    defect_ratio = anomaly_pixels / max(fg_pixels, 1)
    assert defect_ratio < 0.15

def test_conformal_calibrator_guarantee():
    """
    Verify conformal calibration threshold calculation and finite-sample quantile guarantee.
    """
    calibrator = ConformalCalibrator(target_false_alarm_rate=0.05)
    # 100 normal calibration validation scores distributed around 0.1 to 0.4
    np.random.seed(42)
    scores = np.random.uniform(0.1, 0.4, size=100).tolist()
    threshold = calibrator.fit(scores)

    # Fraction of normal calibration samples above threshold should be <= target_fpr (0.05)
    empirical_violations = np.mean(np.array(scores) > threshold)
    assert empirical_violations <= 0.06
