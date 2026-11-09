# VISTA-VLM v4.0 REST API Reference

The VISTA-VLM FastAPI backend provides a high-throughput, fully typed REST API for industrial inspection, batch evaluation, benchmark reporting, and inspector feedback.

**Default Base URL**: `http://localhost:8000`

---

## Endpoints

### 1. `GET /health`
Returns system status, active execution device, backbone architecture, and calibrated categories.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "system_version": "4.0.0",
  "device": "cpu",
  "backbone": "dinov2_vits14",
  "models_loaded": true,
  "calibrated_categories": ["bottle"]
}
```

---

### 2. `GET /health/live` (Kubernetes Liveness Probe)
Checks whether the FastAPI application process is alive and responsive. Returns immediately without waiting on hardware or model resources.

**Response `200 OK`**:
```json
{
  "status": "live",
  "timestamp": "2026-11-09T14:00:00"
}
```

---

### 3. `GET /health/ready` (Kubernetes Readiness Probe)
Verifies that foundation models, conformal thresholds, and OOD detection matrices are loaded into memory and ready to serve inspection traffic.

**Response `200 OK`**:
```json
{
  "status": "ready",
  "models_initialized": true,
  "calibrator_loaded": true,
  "ood_detector_loaded": true,
  "timestamp": "2026-11-09T14:00:00"
}
```

---

### 4. `GET /model/info`
Returns architectural metadata, parameter counts, and calibration parameters.

**Response `200 OK`**:
```json
{
  "architecture": "VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)",
  "version": "4.0.0",
  "backbone": "dinov2_vits14",
  "embed_dim": 384,
  "patch_size": 14,
  "grid_size": 37,
  "parameters_total": 24124340,
  "parameters_trainable": 2073908,
  "language_verifier": "CLIP ViT-B/16 (Frozen Multimodal Ranker with Text Cache)",
  "calibration_method": "split_conformal",
  "supported_categories": ["bottle", "cable", "leather", "capsule", "tile", "candle"]
}
```

---

### 5. `POST /inspect`
Executes end-to-end visual anomaly detection, localization, and multimodal verification on a single image.

**Request**:
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image binary (PNG, JPEG, BMP, TIFF, WEBP; max 20MB)
  - `category`: Target category name (e.g. `bottle`, `cable`)

**Security & Validation**:
- Inspects binary magic bytes. Returns `415 Unsupported Media Type` if the payload does not match valid image headers.
- Enforces 20MB file size limits.

**Response `200 OK`**:
```json
{
  "inspection_id": "c71a39bf-90e6-4277-a06f-702fe6198f79",
  "timestamp": "2026-11-09T14:00:00",
  "category": "bottle",
  "decision": {
    "is_anomaly": true,
    "confidence_level": 0.95,
    "conformal_p_value": 0.0124,
    "autonomous_action": "ANOMALY_DETECTED",
    "requires_human_review": false,
    "escalation_reason": null
  },
  "metrics": {
    "anomaly_score": 0.7421,
    "calibrated_threshold": 0.3962,
    "raw_score": 0.7421,
    "ood_distance": 0.000412,
    "is_ood": false,
    "defect_area_percentage": 2.45
  },
  "localization": {
    "has_defect_regions": true,
    "num_regions": 1,
    "bounding_boxes": [
      {
        "box_id": 1,
        "x_min": 180,
        "y_min": 210,
        "x_max": 255,
        "y_max": 310,
        "area_pixels": 7500,
        "mean_anomaly_score": 0.8124,
        "max_anomaly_score": 0.9412
      }
    ],
    "heatmap_base64": "data:image/png;base64,...",
    "mask_base64": "data:image/png;base64,..."
  },
  "language_verification": {
    "verified": true,
    "defect_candidate": "broken glass",
    "confidence": 0.894,
    "grounded_explanation": "Inspection of bottle reveals anomaly score of 0.7421 (calibrated threshold: 0.3962). Detected 1 localized defect region(s) covering 2.45% of total surface area. Primary anomaly located at bounding box (180, 210, 255, 310) with peak intensity 0.9412.",
    "is_ambiguous": false
  },
  "latency_breakdown_ms": {
    "feature_extraction": 42.1,
    "fusion_and_decoding": 18.4,
    "calibration_and_ood": 1.2,
    "language_verification": 1.8,
    "total_ms": 63.5
  }
}
```

---

### 6. `POST /inspect/batch`
Executes inspection across a batch of images (maximum 50 items per payload).

---

### 7. `GET /benchmark/results`
Serves current 4-track benchmark metrics and category evaluations to the frontend dashboard.

---

### 8. `POST /feedback`
Appends inspector annotations and review decisions to an immutable audit ledger (`benchmarks/feedback/*.json`).
