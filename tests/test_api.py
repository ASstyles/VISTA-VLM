import pytest
import io
import json
from PIL import Image
from fastapi import UploadFile
from vista_vlm.api.routes import (
    health_check,
    get_model_info,
    get_benchmark_results,
    submit_feedback,
    inspect_image
)
from vista_vlm.api.schemas import FeedbackRequest

def test_api_health():
    res = health_check()
    assert res.status == "healthy"
    assert res.system_version == "1.0.0"
    assert res.backbone == "dinov2_vits14"

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
    # Create small test image
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
