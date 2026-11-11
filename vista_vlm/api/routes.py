import os
import time
import json
import uuid
import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
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

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB limit
CHUNK_SIZE = 64 * 1024  # 64KB chunks


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
    """General health check."""
    pipeline = get_pipeline()
    calibrated_cats = list(pipeline.calibrator.category_thresholds.keys())
    model_st = getattr(pipeline.model, "state", "READY")
    if hasattr(model_st, "value"):
        model_st = model_st.value
    return HealthResponse(
        status="healthy",
        system_version="6.0.0",
        device=str(pipeline.device),
        backbone=pipeline.config.model.backbone,
        models_loaded=True,
        model_state=str(model_st),
        calibrated_categories=calibrated_cats if calibrated_cats else ["bottle (default)"]
    )


@router.get("/health/live")
def liveness_check():
    """Kubernetes liveness probe: returns 200 immediately if process is alive."""
    return {"status": "alive", "timestamp": time.time(), "version": "6.0.0"}


@router.get("/health/ready")
def readiness_check():
    """Kubernetes readiness probe: verifies pipeline, model, calibrator, and OOD modules are ready to serve."""
    try:
        pipeline = get_pipeline()
        is_ready = bool(
            pipeline.model is not None
            and getattr(pipeline.model, "is_ready", True)
            and pipeline.calibrator is not None
        )
        if not is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model pipeline is initializing")

        model_st = getattr(pipeline.model, "state", "READY")
        if hasattr(model_st, "value"):
            model_st = model_st.value

        return {
            "status": "ready",
            "version": "6.0.0",
            "model_state": str(model_st),
            "device": str(pipeline.device),
            "calibrated_categories": list(pipeline.calibrator.category_thresholds.keys()),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service not ready: {str(e)}")


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    pipeline = get_pipeline()
    footprint = get_model_footprint(pipeline.model)
    return ModelInfoResponse(
        version="6.0.0",
        architecture="VISTA-VLM v6.0 (Vision Intelligence through Spatial and Textual Alignment)",
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


def read_upload_stream_safely(file: UploadFile, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Reads uploaded file stream in bounded chunks to prevent heap memory exhaustion DoS."""
    chunks = []
    total_read = 0
    while True:
        chunk = file.file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file payload exceeds maximum allowed cap of {max_bytes // (1024 * 1024)}MB"
            )
        chunks.append(chunk)

    if total_read == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file uploaded"
        )
    return b"".join(chunks)


def validate_and_decode_image(contents: bytes) -> Image.Image:
    """
    Validates magic bytes, decodes via Pillow with decompression bomb protection,
    and enforces dimensional boundaries.
    """
    if not validate_image_format(contents):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Allowed formats: PNG, JPEG, BMP, TIFF, WEBP."
        )

    # Set Pillow decompression bomb threshold (max 25 Megapixels)
    Image.MAX_IMAGE_PIXELS = 25_000_000
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image dimensions exceed maximum allowed decompression limits (potential decompression bomb)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed image binary could not be parsed: {str(e)}"
        )

    w, h = pil_img.size
    if w < 16 or h < 16 or w > 4096 or h > 4096:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image dimensions ({w}x{h}) outside permissible operating boundaries [16x16 to 4096x4096]."
        )
    return pil_img


def validate_safe_path(path: str, allowed_root: str = "data") -> str:
    """
    Validates that a filesystem path is strictly confined within an authorized sandboxed root.
    Enforces canonical path resolution, realpath containment, symlink dereferencing checks,
    file-type validation, and file-size constraints.
    """
    real_allowed = os.path.realpath(allowed_root)
    real_path = os.path.realpath(path)

    try:
        common_real = os.path.commonpath([real_path, real_allowed])
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-drive or absolute root filesystem path access is strictly forbidden."
        )

    if common_real != real_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Path traversal or symlink escape detected: Access outside {allowed_root}/ is forbidden."
        )

    if not os.path.exists(real_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Specified asset not found in sandboxed repository: {os.path.basename(path)}"
        )

    if not os.path.isfile(real_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specified path is a directory, not a valid image file."
        )

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    if not real_path.lower().endswith(valid_exts):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image file extension. Allowed: PNG, JPEG, BMP, TIFF, WEBP."
        )

    sz = os.path.getsize(real_path)
    if sz > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Referenced image file exceeds maximum cap of {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
        )

    return real_path


@router.post("/inspect", response_model=InspectionResponse)
def inspect_image(
    file: UploadFile = File(...),
    category: str = Form("bottle")
):
    try:
        pipeline = get_pipeline()
        contents = read_upload_stream_safely(file)
        pil_img = validate_and_decode_image(contents)

        result = pipeline.inspect(pil_img, category=category)
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
            safe_p = validate_safe_path(path)
            res = pipeline.inspect(safe_p, category=payload.category)
            results.append(res)
        except HTTPException:
            raise
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
