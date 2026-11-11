import pytest
import io
import json
from PIL import Image
from fastapi import UploadFile, HTTPException
from vista_vlm.api.routes import (
    health_check,
    liveness_check,
    readiness_check,
    get_model_info,
    get_benchmark_results,
    submit_feedback,
    inspect_image
)
from vista_vlm.api.schemas import FeedbackRequest

def test_api_health():
    res = health_check()
    assert res.status == "healthy"
    assert res.system_version in ["4.0.0", "5.0.0"]
    assert res.backbone == "dinov2_vits14"

def test_api_liveness_readiness():
    live = liveness_check()
    assert live["status"] == "alive"

    ready = readiness_check()
    assert ready["status"] == "ready"
    assert "calibrated_categories" in ready

def test_api_model_info():
    res = get_model_info()
    assert "VISTA-VLM" in res.architecture
    assert res.backbone == "dinov2_vits14"
    assert res.parameters_trainable > 0

def test_api_benchmark_results():
    res = get_benchmark_results()
    assert res.status_code == 200
    data = json.loads(res.body.decode("utf-8"))
    assert "baselines_comparison" in data
    assert "ablation_study" in data

def test_api_inspect():
    img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_file = UploadFile(filename="test.png", file=buf)
    res = inspect_image(file=upload_file, category="bottle")
    assert res.inspection_id is not None
    assert res.status is not None
    assert 0.0 <= res.image_anomaly_score <= 1.0
    assert res.reliability.conformal_p_value is not None

def test_api_inspect_invalid_format():
    buf = io.BytesIO(b"not_an_image_binary_data")
    upload_file = UploadFile(filename="corrupt.txt", file=buf)
    with pytest.raises(HTTPException) as excinfo:
        inspect_image(file=upload_file, category="bottle")
    assert excinfo.value.status_code == 415

def test_api_feedback():
    payload = FeedbackRequest(
        inspection_id="test_123",
        category="bottle",
        user_label="normal",
        comments="Inspected manually and confirmed normal."
    )
    res = submit_feedback(payload)
    assert res.status == "recorded"
    assert res.feedback_id is not None


def test_api_batch_path_traversal_rejection():
    from vista_vlm.api.routes import inspect_batch
    from vista_vlm.api.schemas import BatchInspectionRequest
    payload = BatchInspectionRequest(
        category="bottle",
        image_paths=["/etc/passwd", "../../../windows/win.ini", "C:\\Windows\\system.ini"]
    )
    with pytest.raises(HTTPException) as excinfo:
        inspect_batch(payload)
    assert excinfo.value.status_code in [403, 404]

