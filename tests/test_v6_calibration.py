"""
Statistical Calibration and Selective Prediction Tests (V6.0).
==============================================================
Validates:
  1. Finite-sample order statistic rank calculation: k = ceil((n + 1) * (1 - alpha))
  2. Strict rejection when k > n (InsufficientSampleSizeError)
  3. Upper-tail conformal p-value calculation: p(x) = (1 + sum(s_i >= s_test)) / (n + 1)
  4. Decision alignment between conformal threshold and p-value
  5. OOD regularized covariance and fail-closed state on uncalibrated queries
  6. Selective predictor tuning on labelled validation sets
  7. Exact Wilson score intervals and Clopper-Pearson upper bounds
"""

import pytest
import numpy as np

from vista_vlm.calibration.conformal import (
    ConformalCalibrator,
    InsufficientSampleSizeError,
    UncalibratedError
)
from vista_vlm.calibration.ood import (
    OODDetector
)
from vista_vlm.calibration.selective import (
    SelectivePredictor,
    OperationalState,
    PhysicalDisposition,
    wilson_score_interval,
    exact_binomial_upper_bound
)


def test_conformal_finite_sample_rank_calculation():
    # n = 99, alpha = 0.05 -> k = ceil(100 * 0.95) = 95
    cal = ConformalCalibrator(target_false_alarm_rate=0.05)
    scores = list(np.linspace(0.01, 0.99, 99))
    threshold = cal.fit(scores, category="bottle")

    # The 95th 1-indexed order statistic in sorted 99 samples is index 94
    sorted_s = sorted(scores)
    assert threshold == sorted_s[94]
    assert cal.is_calibrated("bottle")


def test_conformal_rejection_when_k_greater_than_n():
    # n = 10, alpha = 0.01 -> k = ceil(11 * 0.99) = ceil(10.89) = 11 > 10
    cal = ConformalCalibrator(target_false_alarm_rate=0.01)
    scores = list(np.linspace(0.1, 1.0, 10))

    with pytest.raises(InsufficientSampleSizeError, match="INSUFFICIENT CALIBRATION DATA"):
        cal.fit(scores, category="bottle")


def test_conformal_p_value_and_decision_consistency():
    cal = ConformalCalibrator(target_false_alarm_rate=0.05)
    scores = list(np.linspace(0.1, 0.9, 100))
    thresh = cal.fit(scores, category="test_cat")

    # Test sample clearly below threshold
    is_anom, p_val, cut = cal.predict(0.2, category="test_cat")
    assert is_anom is False
    assert p_val > 0.05
    assert cut == round(thresh, 6)

    # Test sample clearly above threshold
    is_anom2, p_val2, _ = cal.predict(0.95, category="test_cat")
    assert is_anom2 is True
    assert p_val2 <= 0.05

    # P-value calculation formula verification
    # For score 0.95, 0 calibration scores are >= 0.95 -> p = (1 + 0) / 101 = 0.0099
    expected_p = round(1.0 / 101.0, 6)
    assert p_val2 == expected_p


def test_conformal_uncalibrated_fails_closed():
    cal = ConformalCalibrator()
    with pytest.raises(UncalibratedError, match="UNCALIBRATED"):
        cal.predict(0.5, category="unknown", strict=True)

    # Non-strict mode returns uncalibrated tuple
    is_anom, p_val, cut = cal.predict(0.5, category="unknown", strict=False)
    assert is_anom is False
    assert p_val is None
    assert cut is None


def test_ood_detector_regularized_covariance_and_fails_closed():
    ood = OODDetector(metric="mahalanobis", threshold_percentile=95.0)
    features = np.random.randn(50, 32)
    ood.fit(features, category="rigid_part")

    assert ood.is_fitted("rigid_part")
    # In-distribution vector
    in_dist_vec = np.mean(features, axis=0)
    is_ood, dist, thresh = ood.compute_distance(in_dist_vec, category="rigid_part")
    assert is_ood is False
    assert dist < thresh

    # Extreme OOD vector
    ood_vec = in_dist_vec + 100.0 * np.random.randn(32)
    is_ood2, dist2, _ = ood.compute_distance(ood_vec, category="rigid_part")
    assert is_ood2 is True
    assert dist2 > thresh

    # Uncalibrated category returns fail-closed state
    is_ood_unk, dist_unk, _ = ood.compute_distance(in_dist_vec, category="unknown_cat", strict=False)
    assert is_ood_unk is True
    assert dist_unk == float("inf")


def test_selective_predictor_labelled_tuning():
    sel = SelectivePredictor()
    # Labelled tuning dataset with both normal (0) and defective (1) items
    normal_scores = list(np.random.normal(0.2, 0.05, 50))
    defect_scores = list(np.random.normal(0.8, 0.05, 50))
    scores = normal_scores + defect_scores
    labels = [0] * 50 + [1] * 50

    delta = sel.calibrate_uncertainty_band(scores, labels, calibrated_threshold=0.5, target_risk=0.01)
    assert 0.001 <= delta <= 0.30

    # Evaluate decision
    res_normal = sel.evaluate_decision(
        anomaly_score=0.1,
        calibrated_threshold=0.5,
        is_ood=False,
        is_calibrated=True
    )
    assert res_normal["state"] == OperationalState.NORMAL.value
    assert res_normal["disposition"] == PhysicalDisposition.ACCEPT.value

    res_anom = sel.evaluate_decision(
        anomaly_score=0.9,
        calibrated_threshold=0.5,
        is_ood=False,
        is_calibrated=True
    )
    assert res_anom["state"] == OperationalState.ANOMALY_DETECTED.value
    assert res_anom["disposition"] == PhysicalDisposition.REJECT.value

    # Uncertain decision inside uncertainty band
    res_unc = sel.evaluate_decision(
        anomaly_score=0.5 + delta * 0.5,
        calibrated_threshold=0.5,
        is_ood=False,
        is_calibrated=True
    )
    assert res_unc["state"] == OperationalState.UNCERTAIN.value
    assert res_unc["disposition"] == PhysicalDisposition.HUMAN_REVIEW.value
    assert res_unc["human_review_required"] is True


def test_wilson_interval_and_exact_binomial_bound():
    # 0 escapes out of 100 defect items
    low, high = wilson_score_interval(0, 100, confidence=0.95)
    assert low == 0.0
    assert 0.03 < high < 0.04  # ~3.6% rule of 3 for 100 trials

    # Exact Clopper-Pearson 95% upper bound
    ub = exact_binomial_upper_bound(0, 100, confidence=0.95)
    assert 0.025 < ub < 0.035

