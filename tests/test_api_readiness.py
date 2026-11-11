"""
Comprehensive Production Readiness Gating Tests for VISTA-VLM API.
==================================================================
Tests all 10 production readiness failure and success scenarios:
  1. All components valid -> HTTP 200, status='ready'
  2. Model missing -> HTTP 503
  3. Model uninitialized -> HTTP 503
  4. Memory bank missing -> HTTP 503
  5. Memory bank empty -> HTTP 503
  6. Calibrator missing -> HTTP 503
  7. Calibration thresholds missing -> HTTP 503
  8. OOD detector missing -> HTTP 503
  9. OOD detector unfitted -> HTTP 503
  10. Invalid artifact (model state invalid) -> HTTP 503
"""

import pytest
from fastapi import HTTPException
from vista_vlm.api.routes import readiness_check
from vista_vlm.models.vista_model import ModelState


def test_readiness_all_components_valid(mock_ready_pipeline):
    """Scenario 1: All components valid -> HTTP 200, status='ready'."""
    res = readiness_check()
    assert res["status"] == "ready"
    assert res["version"] == "7.0.0"
    assert res["model_state"] == "READY"
    assert res["calibrated_categories"] == ["bottle"]
    assert "timestamp" in res


def test_readiness_model_missing(mock_pipeline_builder):
    """Scenario 2: Model missing -> HTTP 503."""
    mock_pipeline_builder(has_model=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "model pipeline is uninitialized" in exc.value.detail.lower()


def test_readiness_model_uninitialized(mock_pipeline_builder):
    """Scenario 3: Model uninitialized (is_ready=False) -> HTTP 503."""
    mock_pipeline_builder(model_is_ready=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "model pipeline is uninitialized" in exc.value.detail.lower()


def test_readiness_memory_bank_missing(mock_pipeline_builder):
    """Scenario 4: Memory bank missing -> HTTP 503."""
    mock_pipeline_builder(has_memory_bank=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "memory bank is not fitted" in exc.value.detail.lower()


def test_readiness_memory_bank_empty(mock_pipeline_builder):
    """Scenario 5: Memory bank empty (memory_fitted=False) -> HTTP 503."""
    mock_pipeline_builder(memory_fitted=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "memory bank is not fitted" in exc.value.detail.lower()


def test_readiness_calibrator_missing(mock_pipeline_builder):
    """Scenario 6: Calibrator missing -> HTTP 503."""
    mock_pipeline_builder(has_calibrator=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "conformal calibrator is not initialized" in exc.value.detail.lower()


def test_readiness_calibration_thresholds_missing(mock_pipeline_builder):
    """Scenario 7: Calibration thresholds empty -> HTTP 503."""
    mock_pipeline_builder(calibrator_thresholds={})
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "conformal calibrator is not initialized" in exc.value.detail.lower()


def test_readiness_ood_detector_missing(mock_pipeline_builder):
    """Scenario 8: OOD detector missing -> HTTP 503."""
    mock_pipeline_builder(has_ood_detector=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "ood detector is not fitted" in exc.value.detail.lower()


def test_readiness_ood_detector_unfitted(mock_pipeline_builder):
    """Scenario 9: OOD detector unfitted (no prototypes) -> HTTP 503."""
    mock_pipeline_builder(ood_fitted=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "ood detector is not fitted" in exc.value.detail.lower()


def test_readiness_invalid_artifact(mock_pipeline_builder):
    """Scenario 10: Invalid artifact / ModelState.INVALID -> HTTP 503."""
    mock_pipeline_builder(model_state=ModelState.INVALID)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "model pipeline is uninitialized or not ready" in exc.value.detail.lower()

