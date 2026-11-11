import pytest
import os
import numpy as np
from PIL import Image
from vista_vlm.config.schema import VISTAConfig
from vista_vlm.models.pipeline import VISTAInspectionPipeline
from vista_vlm.api.schemas import InspectionResponse

def test_end_to_end_pipeline_execution():
    cfg = VISTAConfig()
    ckpt = "checkpoints/vista_mvtec_bottle.pth" if os.path.exists("checkpoints/vista_mvtec_bottle.pth") else None
    calib = "checkpoints/calibration_mvtec.json" if os.path.exists("checkpoints/calibration_mvtec.json") else None
    ood = "checkpoints/ood_mvtec.json" if os.path.exists("checkpoints/ood_mvtec.json") else None

    pipeline = VISTAInspectionPipeline(
        config=cfg,
        checkpoint_path=ckpt,
        calibration_path=calib,
        ood_path=ood
    )

    # Test image
    img = Image.new("RGB", (256, 256), color=(180, 180, 180))
    res = pipeline.inspect(img, category="bottle")

    assert isinstance(res, InspectionResponse)
    assert res.inspection_id is not None
    assert res.category == "bottle"
    assert res.status in ["NORMAL", "ANOMALOUS", "ANOMALY_DETECTED", "UNCERTAIN", "OOD", "ESCALATE_TO_HUMAN"]
    assert 0.0 <= res.image_anomaly_score <= 1.0
    assert res.reliability.calibrated_threshold > 0.0
    assert res.heatmap_base64 is not None
    assert res.mask_base64 is not None
