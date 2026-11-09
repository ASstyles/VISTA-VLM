import pytest
import numpy as np
import tempfile
import os
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
    np.random.seed(42)
    detector = OODDetector(metric="cosine", threshold_percentile=95.0)

    # In-distribution prototype cluster centered around non-zero visual embedding centroid
    proto_features = np.ones((50, 64)) * 2.0 + np.random.randn(50, 64) * 0.2
    detector.fit(proto_features, category="bottle")

    # In-distribution query: close to cluster centroid
    in_dist = proto_features.mean(axis=0) + np.random.randn(64) * 0.01
    is_ood_in, dist_in, _ = detector.compute_distance(in_dist, category="bottle")
    assert not is_ood_in

    # Out-of-distribution query
    out_dist = -proto_features[0] * 5.0
    is_ood_out, dist_out, _ = detector.compute_distance(out_dist, category="bottle")
    assert dist_out > dist_in

def test_ood_mahalanobis_serialization_roundtrip():
    """Validates SEC-13 fix: cov_inv must survive save and load without reverting to cosine distance."""
    detector = OODDetector(metric="mahalanobis", threshold_percentile=98.0)
    features = np.random.randn(40, 32) + 2.0
    detector.fit(features, category="cable")

    test_feat = features[0] + 0.1
    is_ood_orig, dist_orig, thresh_orig = detector.compute_distance(test_feat, category="cable")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        detector.save(tmp_path)
        restored = OODDetector()
        restored.load(tmp_path)

        assert "cable" in restored.cov_inv
        is_ood_rest, dist_rest, thresh_rest = restored.compute_distance(test_feat, category="cable")
        assert np.isclose(dist_orig, dist_rest, atol=1e-4)
        assert np.isclose(thresh_orig, thresh_rest, atol=1e-4)
        assert is_ood_orig == is_ood_rest
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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

def test_selective_predictor_risk_coverage():
    predictor = SelectivePredictor(uncertainty_band_width=0.05, enabled=True)
    scores = np.array([0.1, 0.15, 0.25, 0.48, 0.52, 0.8, 0.9])
    labels = np.array([0,   0,    0,    0,    1,    1,   1])
    thresh = 0.50

    risk_eval = predictor.evaluate_false_pass_risk(scores, labels, calibrated_threshold=thresh)
    assert "observed_false_pass_rate" in risk_eval
    assert "observed_false_alarm_rate" in risk_eval
    assert "false_pass_ci_95" in risk_eval
    assert "coverage" in risk_eval
    assert risk_eval["coverage"] < 1.0 # Some samples are in band [0.45, 0.55]

    curve = predictor.compute_risk_coverage_curve(scores, labels, calibrated_threshold=thresh, num_steps=10)
    assert len(curve) == 10
    assert "delta" in curve[0]
    assert "coverage" in curve[0]
