import os
import sys
import time
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse

from vista_vlm.api.schemas import (
    InspectionResponse,
    BatchInspectionRequest,
    BatchInspectionResponse,
    HealthResponse,
    ModelInfoResponse,
    FeedbackRequest,
    FeedbackResponse
)
from vista_vlm.optimization.exporter import get_model_footprint
from vista_vlm.config.schema import VISTAConfig
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.api")

router = APIRouter()

# Global pipeline instance initialized lazily
pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        from vista_vlm.models.pipeline import VISTAInspectionPipeline
        cfg = VISTAConfig()
        ckpt = "checkpoints/vista_mvtec_bottle.pth" if os.path.exists("checkpoints/vista_mvtec_bottle.pth") else None
        calib = "checkpoints/calibration_mvtec.json" if os.path.exists("checkpoints/calibration_mvtec.json") else None
        ood = "checkpoints/ood_mvtec.json" if os.path.exists("checkpoints/ood_mvtec.json") else None
        pipeline_instance = VISTAInspectionPipeline(
            config=cfg,
            checkpoint_path=ckpt,
            calibration_path=calib,
            ood_path=ood,
        )
    return pipeline_instance


@router.get("/health", response_model=HealthResponse)
def health_check():
    """General health check (backward compatible)."""
    pipeline = get_pipeline()
    calibrated_cats = list(pipeline.calibrator.category_thresholds.keys())
    return HealthResponse(
        status="healthy",
        system_version="4.0.0",
        device=str(pipeline.device),
        backbone=pipeline.config.model.backbone,
        models_loaded=True,
        calibrated_categories=calibrated_cats if calibrated_cats else ["bottle (default)"]
    )


@router.get("/health/live")
def liveness_check():
    """Kubernetes liveness probe: returns 200 immediately if process is alive."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/health/ready")
def readiness_check():
    """Kubernetes readiness probe: verifies pipeline, model, calibrator, and OOD modules are ready to serve."""
    try:
        pipeline = get_pipeline()
        is_ready = bool(pipeline.model is not None and pipeline.calibrator is not None)
        if not is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model pipeline is initializing")
        return {
            "status": "ready",
            "device": str(pipeline.device),
            "calibrated_categories": list(pipeline.calibrator.category_thresholds.keys()),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service not ready: {str(e)}")


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    pipeline = get_pipeline()
    footprint = get_model_footprint(pipeline.model)
    return ModelInfoResponse(
        architecture="VISTA-VLM v4.0 (Vision Intelligence through Spatial and Textual Alignment)",
        backbone=pipeline.config.model.backbone,
        embed_dim=pipeline.config.model.embed_dim,
        patch_size=pipeline.config.data.patch_size,
        grid_size=pipeline.config.data.grid_size,
        parameters_total=footprint["parameters_total"],
        parameters_trainable=footprint["parameters_trainable"],
        language_verifier="CLIP ViT-B/16 (Cached Multimodal Ranker)",
        calibration_method=pipeline.config.calibration.method,
        supported_categories=["bottle", "cable", "leather", "capsule", "tile", "candle", "pcb1"]
    )


def validate_image_format(contents: bytes) -> bool:
    """Validates image magic bytes to ensure valid image payload."""
    if len(contents) < 4:
        return False
    # PNG: \x89PNG
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    # JPEG: \xff\xd8\xff
    if contents.startswith(b"\xff\xd8\xff"):
        return True
    # BMP: BM
    if contents.startswith(b"BM"):
        return True
    # TIFF: II*\x00 or MM\x00*
    if contents.startswith(b"II*\x00") or contents.startswith(b"MM\x00*"):
        return True
    # WEBP: RIFF....WEBP
    if contents.startswith(b"RIFF") and len(contents) >= 12 and contents[8:12] == b"WEBP":
        return True
    return False


@router.post("/inspect", response_model=InspectionResponse)
def inspect_image(
    file: UploadFile = File(...),
    category: str = Form("bottle")
):
    try:
        pipeline = get_pipeline()
        contents = file.file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file uploaded")

        # Size check (max 20MB)
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File too large (exceeds 20MB limit)")

        # Format validation
        if not validate_image_format(contents):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported image format. Allowed formats: PNG, JPEG, BMP, TIFF, WEBP."
            )

        result = pipeline.inspect(contents, category=category)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inspection: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal inspection pipeline error")


@router.post("/inspect/batch", response_model=BatchInspectionResponse)
def inspect_batch(payload: BatchInspectionRequest):
    pipeline = get_pipeline()
    t0 = time.time()
    results = []

    # Limit batch size to 50
    if len(payload.image_paths) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch size exceeds maximum limit of 50 images")

    for path in payload.image_paths:
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")
            res = pipeline.inspect(path, category=payload.category)
            results.append(res)
        except Exception as e:
            logger.warning(f"Batch item failed for {path}: {e}")

    total_time = round((time.time() - t0) * 1000.0, 2)
    return BatchInspectionResponse(
        total_images=len(results),
        inspections=results,
        total_execution_time_ms=total_time
    )


@router.get("/benchmark/results")
def get_benchmark_results():
    results = {}
    base_file = "benchmarks/results/baselines_comparison.json"
    ablation_file = "benchmarks/results/ablation_study.json"
    vista_file = "benchmarks/results/vista_results_bottle.json"
    multi_track_file = "benchmarks/results/multi_track_benchmark.json"

    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            results["baselines_comparison"] = json.load(f)
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            results["ablation_study"] = json.load(f)
    if os.path.exists(vista_file):
        with open(vista_file, "r") as f:
            results["vista_detailed_bottle"] = json.load(f)
    if os.path.exists(multi_track_file):
        with open(multi_track_file, "r") as f:
            results["multi_track_benchmark"] = json.load(f)

    return JSONResponse(content=results)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest):
    feedback_id = f"fb_{uuid.uuid4().hex[:10]}"
    fb_record = {
        "feedback_id": feedback_id,
        "inspection_id": payload.inspection_id,
        "category": payload.category,
        "user_label": payload.user_label,
        "comments": payload.comments,
        "correct_defect_type": payload.correct_defect_type,
        "timestamp": time.time()
    }

    fb_dir = "benchmarks/feedback"
    os.makedirs(fb_dir, exist_ok=True)
    fb_file = os.path.join(fb_dir, f"{feedback_id}.json")
    with open(fb_file, "w") as f:
        json.dump(fb_record, f, indent=2)

    return FeedbackResponse(
        status="recorded",
        message="Human-review feedback stored separately from model predictions.",
        feedback_id=feedback_id
    )
