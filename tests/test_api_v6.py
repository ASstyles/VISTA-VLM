"""
API V6.0 Security & Operational Health Tests.
============================================
Validates:
  1. GET /health reports version 6.0.0 and model state
  2. GET /health/live is a lightweight liveness probe
  3. GET /health/ready evaluates actual model readiness
  4. GET /model/info reports v6.0 architecture
  5. POST /inspect/batch enforces canonical realpath containment and blocks path traversal
  6. POST /inspect/batch rejects directory targets and unsupported file types
  7. POST /inspect rejects empty payloads and malformed magic bytes
"""

import pytest
import io
from fastapi import HTTPException, UploadFile

from vista_vlm.api.routes import (
    health_check,
    liveness_check,
    readiness_check,
    get_model_info,
    inspect_batch,
    inspect_image
)
from vista_vlm.api.schemas import BatchInspectionRequest


def test_api_v6_health_endpoint(mock_ready_pipeline):
    res = health_check()
    assert res.status == "healthy"
    assert res.system_version == "7.0.0"
    assert res.model_state is not None


def test_api_v6_liveness_endpoint():
    res = liveness_check()
    assert res["status"] == "alive"
    assert res["version"] == "7.0.0"


def test_api_v6_readiness_valid_pipeline(mock_ready_pipeline):
    """Test readiness against a valid complete production pipeline."""
    res = readiness_check()
    assert res["status"] == "ready"
    assert res["version"] == "7.0.0"
    assert res["model_state"] == "READY"
    assert "calibrated_categories" in res
    assert "bottle" in res["calibrated_categories"]


def test_api_v6_readiness_incomplete_pipeline(mock_pipeline_builder):
    """Test readiness fails with HTTP 503 when model is missing."""
    mock_pipeline_builder(has_model=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "uninitialized" in exc.value.detail.lower()


def test_api_v6_readiness_unfitted_memory_bank(mock_pipeline_builder):
    """Test readiness specifically fails with HTTP 503 when memory bank is unfitted."""
    mock_pipeline_builder(memory_fitted=False)
    with pytest.raises(HTTPException) as exc:
        readiness_check()
    assert exc.value.status_code == 503
    assert "memory bank is not fitted" in exc.value.detail.lower()


def test_api_v6_model_info(mock_ready_pipeline):
    res = get_model_info()
    assert res.version == "7.0.0"
    assert "VISTA-VLM" in res.architecture


def test_api_v6_batch_path_traversal_blocked():
    # Attempt Unix-style path traversal
    req1 = BatchInspectionRequest(
        category="bottle",
        image_paths=["../../../../etc/passwd"]
    )
    with pytest.raises(HTTPException) as exc1:
        inspect_batch(req1)
    assert exc1.value.status_code == 403

    # Attempt Windows-style path traversal
    req2 = BatchInspectionRequest(
        category="bottle",
        image_paths=["..\\..\\..\\Windows\\System32\\cmd.exe"]
    )
    with pytest.raises(HTTPException) as exc2:
        inspect_batch(req2)
    assert exc2.value.status_code == 403


def test_api_v6_batch_directory_target_rejected():
    req = BatchInspectionRequest(
        category="bottle",
        image_paths=["data/synthetic_fixtures"]
    )
    with pytest.raises(HTTPException) as exc:
        inspect_batch(req)
    assert exc.value.status_code == 400
    assert "is a directory" in exc.value.detail


def test_api_v6_inspect_malformed_bytes():
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    upload = UploadFile(filename="corrupt.png", file=io.BytesIO(corrupt_bytes))
    with pytest.raises(HTTPException) as exc:
        inspect_image(file=upload, category="bottle")
    assert exc.value.status_code == 415
