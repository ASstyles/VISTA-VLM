import os
import sys
import io
import time
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import UploadFile
from vista_vlm.api.routes import (
    health_check,
    get_model_info,
    get_benchmark_results,
    submit_feedback,
    inspect_image,
    get_pipeline
)
from vista_vlm.api.schemas import FeedbackRequest

def run_tests():
    print("Testing VISTA-VLM API Endpoints synchronously...")
    t0 = time.time()

    # Preload pipeline
    print("1. Loading VISTA Pipeline...")
    _ = get_pipeline()
    print("   Pipeline successfully loaded.")

    # 2. Health Check
    print("2. Testing /health...")
    res_health = health_check()
    print(f"   [GET /health] Status: {res_health.status} | Backbone: {res_health.backbone} | Calibrated: {res_health.calibrated_categories}")
    assert res_health.status == "healthy"

    # 3. Model Info
    print("3. Testing /model/info...")
    res_info = get_model_info()
    print(f"   [GET /model/info] Total params: {res_info.parameters_total:,} | Trainable: {res_info.parameters_trainable:,}")
    assert res_info.parameters_trainable > 0

    # 4. Benchmark Results
    print("4. Testing /benchmark/results...")
    res_bench = get_benchmark_results()
    bench_data = res_bench.body.decode("utf-8")
    assert "baselines_comparison" in bench_data
    print("   [GET /benchmark/results] Baselines and ablation studies loaded.")

    # 5. Feedback
    print("5. Testing /feedback...")
    fb_req = FeedbackRequest(
        inspection_id="test_id_999",
        category="bottle",
        user_label="normal",
        comments="Verified normal surface."
    )
    res_fb = submit_feedback(fb_req)
    print(f"   [POST /feedback] Status: {res_fb.status} | Feedback ID: {res_fb.feedback_id}")
    assert res_fb.status == "recorded"

    # 6. Single Inspection
    print("6. Testing /inspect...")
    test_img_path = "data/processed/mvtec/bottle/test/broken_large/000.png"
    if os.path.exists(test_img_path):
        with open(test_img_path, "rb") as f:
            file_bytes = f.read()
    else:
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        file_bytes = buf.getvalue()

    upload_file = UploadFile(filename="sample_bottle.png", file=io.BytesIO(file_bytes))
    res_insp = inspect_image(file=upload_file, category="bottle")
    print(f"   [POST /inspect] Status: {res_insp.status}")
    print(f"     Score: {res_insp.image_anomaly_score}")
    print(f"     Threshold: {res_insp.reliability.calibrated_threshold}")
    print(f"     Conformal p-value: {res_insp.reliability.conformal_p_value}")
    print(f"     Interpretation: {res_insp.textual_interpretation}")
    print(f"     Heatmap generated: {res_insp.heatmap_base64 is not None}")
    print(f"     Mask generated: {res_insp.mask_base64 is not None}")
    assert res_insp.image_anomaly_score is not None

    total_time = round(time.time() - t0, 2)
    print(f"\nAll VISTA-VLM API Endpoints tested and fully verified in {total_time}s!")

if __name__ == "__main__":
    run_tests()
