# VISTA-VLM REST API Reference

The VISTA-VLM FastAPI backend provides fully typed REST endpoints for industrial inspection, batch evaluation, benchmark reporting, and inspector feedback.

**Default Base URL**: `http://localhost:8000`

---

## Endpoints

### 1. `GET /health`
Returns system status, active execution device, backbone architecture, and calibrated categories.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "system_version": "1.0.0",
  "device": "cpu",
  "backbone": "dinov2_vits14",
  "models_loaded": true,
  "calibrated_categories": ["bottle"]
}
```

---

### 2. `GET /model/info`
Returns architectural metadata, parameter counts, and calibration methodology.

**Response `200 OK`**:
```json
{
  "architecture": "VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)",
  "backbone": "dinov2_vits14",
  "embed_dim": 384,
  "patch_size": 14,
  "grid_size": 37,
  "parameters_total": 23997394,
  "parameters_trainable": 1950418,
  "language_verifier": "CLIP ViT-B/16 (Frozen Multimodal Ranker)",
  "calibration_method": "split_conformal",
  "supported_categories": ["bottle", "cable", "leather", "capsule", "tile", "candle", "pcb1"]
}
```

---

### 3. `POST /inspect`
Executes end-to-end visual anomaly detection, localization, and multimodal verification on a single image.

**Request**:
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image file (PNG, JPG, BMP up to 20MB)
  - `category`: Target category name (e.g. `bottle`, `cable`)

**Response `200 OK`**:
```json
{
  "inspection_id": "insp_6e919ef7",
  "timestamp": "2026-11-07T13:20:57.123456",
  "category": "bottle",
  "status": "ANOMALOUS",
  "image_anomaly_score": 0.6842,
  "is_anomaly": true,
  "threshold": 0.4954,
  "num_defects_found": 1,
  "bounding_boxes": [
    {
      "x1": 210,
      "y1": 180,
      "x2": 340,
      "y2": 310,
      "confidence": 0.88,
      "label": "broken_large"
    }
  ],
  "region_findings": [
    {
      "box": [210, 180, 340, 310],
      "area": 16900,
      "spatial_anomaly_score": 0.842,
      "top_defect_label": "broken_large",
      "top_similarity_score": 0.784,
      "confidence_margin": 0.142,
      "is_ambiguous": false,
      "candidate_rankings": [
        { "label": "broken_large", "score": 0.784, "prompt": "a photo of a broken bottle" }
      ]
    }
  ],
  "textual_interpretation": "Localized structural defect at bottle mouth with high confidence margin (0.142).",
  "reliability": {
    "calibrated_threshold": 0.4954,
    "conformal_p_value": 0.021,
    "target_false_alarm_rate": 0.05,
    "is_ood": false,
    "ood_distance": 0.241,
    "ood_threshold": 0.850,
    "in_uncertainty_band": false,
    "human_review_recommended": false,
    "review_reason": "Defect score significantly exceeds conformal threshold."
  },
  "heatmap_base64": "iVBORw0KGgo...",
  "mask_base64": "iVBORw0KGgo..."
}
```

---

### 4. `POST /inspect/batch`
Batch evaluation over multiple local image paths.

**Request Body**:
```json
{
  "image_paths": [
    "data/processed/mvtec/bottle/test/broken_large/000.png",
    "data/processed/mvtec/bottle/test/good/000.png"
  ],
  "category": "bottle"
}
```

---

### 5. `GET /benchmark/results`
Returns genuine baseline comparison results and ablation study metrics directly from verified JSON artifacts.

---

### 6. `POST /feedback`
Records inspector verification decisions and manual corrections.

**Request Body**:
```json
{
  "inspection_id": "insp_6e919ef7",
  "category": "bottle",
  "user_label": "anomalous",
  "comments": "Surface crack verified under microscope.",
  "correct_defect_type": "broken_large"
}
```

**Response `200 OK`**:
```json
{
  "status": "recorded",
  "message": "Human-review feedback stored separately from model predictions.",
  "feedback_id": "fb_54737e9623"
}
```
