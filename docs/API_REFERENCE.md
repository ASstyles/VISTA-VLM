# VISTA-VLM V6.0 Industrial REST API Specification

**Document ID:** `V6-SPEC-API-001`  
**Version:** 6.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Overview & Service Architecture

The VISTA-VLM FastAPI backend provides a high-throughput, asynchronous REST API for industrial inspection, batch processing, model lifecycle monitoring, and operator audit telemetry.

- **Base URL**: `http://localhost:8000`
- **Specification**: OpenAPI 3.1.0 (`/docs` and `/redoc`)
- **Transport Security**: TLS 1.3 / HTTPS recommended in production
- **Application Entry Point**: `vista_vlm.api.app:app`

---

## 2. Health & Orchestration Endpoints

### 2.1 `GET /health/live` (Kubernetes Liveness Probe)
Immediate process liveness probe. Returns HTTP 200 if ASGI event loop is active. Does not perform blocking model inference.

**Response `200 OK`**:
```json
{
  "status": "alive",
  "timestamp": 1794420000.0,
  "version": "6.0.0"
}
```

### 2.2 `GET /health/ready` (Kubernetes Readiness Probe)
Evaluates whether model weights, nominal memory bank, and calibration artifacts are loaded and ready to serve production requests.

**Response `200 OK`**:
```json
{
  "status": "ready",
  "version": "6.0.0",
  "model_ready": true,
  "model_state": "READY",
  "category": "bottle",
  "memory_bank_fitted": true,
  "calibrated_categories": ["bottle"]
}
```

### 2.3 `GET /health` (General System Diagnostics)
Returns system status, active execution device, model state, and calibrated category profiles.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "system_version": "6.0.0",
  "device": "cpu",
  "backbone": "dinov2_vits14",
  "models_loaded": true,
  "model_state": "READY",
  "calibrated_categories": ["bottle"]
}
```

---

## 3. Inspection Endpoints

### 3.1 `POST /inspect` (Single Part Inspection)
Ingests an image stream, verifies integrity, executes inference, and returns deterministic operational states and physical dispositions.

**Security & Quota Constraints**:
- Maximum payload: $20\text{ MB}$ (`MAX_IMAGE_BYTES`). Payloads $> 20\text{ MB}$ return `HTTP 413`.
- Dimension bounds: $\text{width}, \text{height} \in [16, 4096]$.
- Raster limit: $25,000,000$ pixels (Decompression bomb defense).
- Magic byte validation: Rejects malformed headers with `HTTP 415`.

**Parameters (Multipart Form-Data)**:
- `file`: Raw image binary (JPEG, PNG, BMP, TIFF, WEBP).
- `category`: Target part category string (e.g. `"bottle"`).

**Response `200 OK`**:
```json
{
  "inspection_id": "insp_20261111_193024_8492",
  "timestamp": "2026-11-11T14:00:24.849Z",
  "category": "bottle",
  "status": "NORMAL",
  "image_anomaly_score": 0.0842,
  "is_anomaly": false,
  "threshold": 0.6876,
  "num_defects_found": 0,
  "bounding_boxes": [],
  "region_findings": [],
  "textual_interpretation": "Inspection verified: No surface defects detected in nominal memory bank comparison.",
  "reliability": {
    "is_calibrated": true,
    "calibrated_threshold": 0.6876,
    "conformal_p_value": 0.9524,
    "target_false_alarm_rate": 0.05,
    "is_ood": false,
    "ood_distance": 1.42,
    "ood_threshold": 4.10,
    "in_uncertainty_band": false,
    "operational_state": "NORMAL",
    "disposition": "ACCEPT",
    "human_review_recommended": false,
    "review_reason": "Decision verified within high confidence region."
  },
  "execution_time_ms": 42.5
}
```

---

### 3.2 `POST /inspect/batch` (Batch Offline Evaluation)
Evaluates multiple filesystem assets. Enforces canonical `realpath` containment to strictly block path traversal and access to system files outside the sandboxed repository root.

**Security Constraints**:
- Enforces `validate_safe_path`: Paths containing `../` escaping `data/` return `HTTP 403 Forbidden`.
- Directory targets return `HTTP 400 Bad Request`.
- Disallowed extensions return `HTTP 415 Unsupported Media Type`.

**Request Body**:
```json
{
  "category": "bottle",
  "image_paths": [
    "data/synthetic_fixtures/mvtec/bottle/test/good/000.png",
    "data/synthetic_fixtures/mvtec/bottle/test/broken_large/000.png"
  ]
}
```

---

### 3.3 `POST /feedback` (Operator Review Telemetry)
Persists operator confirmation or correction into a tamper-evident audit log with SHA-256 integrity digests.

**Request Body**:
```json
{
  "inspection_id": "insp_20261111_193024_8492",
  "category": "bottle",
  "user_label": "normal"
}
```
