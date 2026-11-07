from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class InspectionStatus(str, Enum):
    NORMAL = "NORMAL"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    OOD = "OOD"
    ERROR = "ERROR"

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    spatial_score: float

class CandidateRanking(BaseModel):
    label: str
    score: float
    prompt: str

class RegionFinding(BaseModel):
    box: List[int] # [x1, y1, x2, y2]
    area: int
    spatial_anomaly_score: float
    top_defect_label: str
    top_similarity_score: float
    confidence_margin: float
    is_ambiguous: bool
    candidate_rankings: List[CandidateRanking] = Field(default_factory=list)

class ReliabilityIndicators(BaseModel):
    calibrated_threshold: float
    conformal_p_value: float
    target_false_alarm_rate: float
    is_ood: bool
    ood_distance: float
    ood_threshold: float
    in_uncertainty_band: bool
    human_review_recommended: bool
    review_reason: str

class InspectionResponse(BaseModel):
    inspection_id: str
    timestamp: str
    category: str
    status: InspectionStatus
    image_anomaly_score: float
    is_anomaly: bool
    threshold: float
    num_defects_found: int
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    region_findings: List[RegionFinding] = Field(default_factory=list)
    textual_interpretation: str
    reliability: ReliabilityIndicators
    heatmap_base64: Optional[str] = None
    mask_base64: Optional[str] = None
    execution_time_ms: float

class BatchInspectionRequest(BaseModel):
    category: str
    image_paths: List[str]

class BatchInspectionResponse(BaseModel):
    total_images: int
    inspections: List[InspectionResponse]
    total_execution_time_ms: float

class HealthResponse(BaseModel):
    status: str
    system_version: str
    device: str
    backbone: str
    models_loaded: bool
    calibrated_categories: List[str]

class ModelInfoResponse(BaseModel):
    architecture: str
    backbone: str
    embed_dim: int
    patch_size: int
    grid_size: int
    parameters_total: int
    parameters_trainable: int
    language_verifier: str
    calibration_method: str
    supported_categories: List[str]

class FeedbackRequest(BaseModel):
    inspection_id: str
    category: str
    user_label: str # 'normal', 'defective', 'ambiguous'
    comments: Optional[str] = None
    correct_defect_type: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
