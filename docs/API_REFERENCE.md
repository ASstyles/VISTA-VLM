# VISTA-VLM V5.0 Industrial REST API Specification

**Document ID:** `V5-SPEC-API-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Overview & Service Architecture

The VISTA-VLM FastAPI backend provides a high-throughput, asynchronous REST API for industrial inspection, batch evaluation, and telemetry monitoring.

- **Base URL**: `http://localhost:8000`
- **Specification**: OpenAPI 3.1.0 (`/docs` and `/redoc`)
- **Transport Security**: TLS 1.3 / HTTPS recommended in production

---

## 2. API Endpoints

### 2.1 `GET /health`
Returns system status, active execution device, backbone architecture, and calibrated categories.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "system_version": "5.0.0",
  "device": "cpu",
  "backbone": "dinov2_vits14",
  "models_loaded": true,
  "calibrated_categories": ["bottle"]
}
```

---

### 2.2 `POST /inspect` (Single Part Inspection)
Ingests an image stream, verifies integrity, executes inference, and returns deterministic operational states and dispositions.

**Security & Quota Constraints**:
- Maximum payload: $20\text{ MB}$ (`MAX_IMAGE_BYTES`). Payloads $> 20\text{ MB}$ return `HTTP 413`.
- Dimension bounds: $\text{width}, \text{height} \in [16, 4096]$.
- Raster limit: $25,000,000$ pixels (Decompression bomb defense).

**Parameters (Multipart Form-Data)**:
- `file`: Raw image binary (JPEG, PNG, BMP, TIFF).
- `category`: Target part category string (e.g. `"bottle"`).
- `threshold`: Optional override threshold (float, default `0.5`).

**Response `200 OK`**:
```json
{
  "inspection_id": "insp_20261111_193024_8492",
  "timestamp": "2026-11-11T14:00:24.849Z",
  "category": "bottle",
  "is_anomaly": false,
  "anomaly_score": 0.0842,
  "calibrated_threshold": 0.5000,
  "reliability": {
    "is_calibrated": true,
    "operational_state": "NORMAL",
    "disposition": "ACCEPT",
    "p_value": 0.8412,
    "calibrated_threshold": 0.5000,
    "ood_distance": 0.4210,
    "is_ood": false
  },
  "localization": {
    "has_defect": false,
    "defect_area_ratio": 0.0,
    "bounding_boxes": [],
    "anomaly_map_base64": null
  },
  "execution_time_ms": 649.0
}
```

---

### 2.3 `POST /inspect/batch` (Local Directory Batch Inspection)
Audits a list of local file paths within white-listed directories.

**Security Constraints (SEC-V5-01)**:
- All paths are sandboxed via `validate_safe_path` (`os.path.commonpath`).
- Traversals (`../`) or unauthorized paths return `HTTP 403 Forbidden`.

**Request Body**:
```json
{
  "image_paths": [
    "data/synthetic_fixtures/mvtec_like/bottle/test/good/000.png",
    "data/synthetic_fixtures/mvtec_like/bottle/test/good/001.png"
  ],
  "category": "bottle"
}
```

**Response `200 OK`**:
```json
{
  "batch_size": 2,
  "successful_inspections": 2,
  "failed_inspections": 0,
  "results": [ ... ],
  "mean_execution_time_ms": 648.5
}
```

---

### 2.4 Health Probes
- **`GET /health/live`**: Kubernetes liveness check. Returns `200 OK` if process is responding.
- **`GET /health/ready`**: Kubernetes readiness check. Verifies models and calibration caches are loaded into memory.
