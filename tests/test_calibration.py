import pytest
import numpy as np
from vista_vlm.calibration.conformal import ConformalCalibrator
from vista_vlm.calibration.ood import OODDetector
from vista_vlm.calibration.selective import SelectivePredictor

def test_split_conformal_calibrator():
    calibrator = ConformalCalibrator(target_false_alarm_rate=0.05)
    
    # 100 normal calibration scores
    normal_scores = np.random.uniform(0.1, 0.4, size=100).tolist()
    threshold = calibrator.fit(normal_scores, category="test_cat")

    # Threshold must be near the 95th percentile
    assert 0.1 <= threshold <= 0.45
    assert "test_cat" in calibrator.category_thresholds

    # Monotonicity of conformal p-values
    is_ano1, p_low, _ = calibrator.predict(score=0.15, category="test_cat")
    is_ano2, p_high, _ = calibrator.predict(score=0.38, category="test_cat")
    assert p_low > p_high # lower score should have higher p-value of being normal
    assert not is_ano1

def test_ood_detector():
    detector = OODDetector(metric="cosine", threshold_percentile=95.0)

    # In-distribution prototype cluster centered around non-zero visual embedding centroid
    proto_features = np.ones((50, 64)) * 2.0 + np.random.randn(50, 64) * 0.2
    detector.fit(proto_features, category="bottle")

    # In-distribution query
    in_dist = proto_features[0:1] + np.random.randn(1, 64) * 0.05
    is_ood_in, dist_in, _ = detector.compute_distance(in_dist[0], category="bottle")
    assert not is_ood_in

    # Out-of-distribution query
    out_dist = -proto_features[0] * 5.0
    is_ood_out, dist_out, _ = detector.compute_distance(out_dist, category="bottle")
    assert dist_out > dist_in

def test_selective_predictor():
    predictor = SelectivePredictor(uncertainty_band_width=0.04, enabled=True)

    # Case 1: High confidence normal
    res_norm = predictor.evaluate_decision(
        anomaly_score=0.20,
        calibrated_threshold=0.50,
        is_ood=False
    )
    assert not res_norm["human_review_required"]
    assert not res_norm["in_uncertainty_band"]

    # Case 2: In uncertainty band [0.46, 0.54]
    res_unc = predictor.evaluate_decision(
        anomaly_score=0.51,
        calibrated_threshold=0.50,
        is_ood=False
    )
    assert res_unc["in_uncertainty_band"]
    assert res_unc["human_review_required"]

    # Case 3: Out-of-distribution
    res_ood = predictor.evaluate_decision(
        anomaly_score=0.20,
        calibrated_threshold=0.50,
        is_ood=True
    )
    assert res_ood["human_review_required"]
