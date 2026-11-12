import numpy as np
import tempfile
import json
import os
import pytest
from vista_vlm.calibration.conformal import ConformalCalibrator, InsufficientSampleSizeError
from vista_vlm.calibration.ood import OODDetector
from vista_vlm.calibration.selective import (
    SelectivePredictor,
    wilson_score_interval,
    exact_binomial_upper_bound
)

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


def test_conformal_insufficient_sample_size():
    calibrator = ConformalCalibrator(target_false_alarm_rate=0.01) # alpha=0.01 requires n >= 99
    small_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    with pytest.raises(InsufficientSampleSizeError):
        calibrator.fit(small_scores, category="insufficient_cat")


def test_conformal_artifact_tamper_rejection():
    calibrator = ConformalCalibrator(target_false_alarm_rate=0.05)
    normal_scores = np.random.uniform(0.1, 0.4, size=50).tolist()
    calibrator.fit(normal_scores, category="secure_cat")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        calibrator.save(tmp_path)
        # Tamper with the saved scores
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["calibration_scores"]["secure_cat"][0] = 99.0 # Tamper!
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        reloaded = ConformalCalibrator()
        with pytest.raises(ValueError, match="TAMPERING/CORRUPTION DETECTED"):
            reloaded.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_selective_predictor_prevents_zero_coverage_on_tight_distribution():
    """Verify that calibrate_uncertainty_band does not collapse to 0% coverage on clustered scores."""
    predictor = SelectivePredictor()
    # Scores clustered tightly around 0.77 (like Cable/Candle)
    tune_scores = np.array([
        0.75, 0.76, 0.762, 0.765, 0.768, 0.770, 0.772,  # Normal (0)
        0.775, 0.778, 0.780, 0.782, 0.785, 0.788, 0.790   # Anomaly (1)
    ])
    tune_labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    thresh = 0.771

    delta = predictor.calibrate_uncertainty_band(
        tune_scores, tune_labels,
        calibrated_threshold=thresh,
        min_coverage=0.50
    )

    # Delta must be reasonably small, not wiping out everything
    assert 0.001 <= delta <= 0.05

    # Evaluate on test set with similar spread
    test_scores = np.array([0.74, 0.755, 0.76, 0.766, 0.775, 0.78, 0.785, 0.795])
    test_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    res = predictor.evaluate_false_pass_risk(test_scores, test_labels, calibrated_threshold=thresh)

    assert res["coverage"] >= 0.50
    assert res["accepted_samples"] > 0
    assert res["false_passes"] == 0


def test_wilson_and_exact_binomial_bounds():
    low, high = wilson_score_interval(0, 10, confidence=0.95)
    assert low == 0.0
    assert 0.20 <= high <= 0.35

    ub = exact_binomial_upper_bound(0, 10, confidence=0.95)
    assert 0.20 <= ub <= 0.35

    # 0 trials fallback
    low0, high0 = wilson_score_interval(0, 0)
    assert low0 == 0.0 and high0 == 1.0
    assert exact_binomial_upper_bound(0, 0) == 1.0
