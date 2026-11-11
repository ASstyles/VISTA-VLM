export type InspectionStatus = 'NORMAL' | 'ANOMALOUS' | 'ANOMALY_DETECTED' | 'ESCALATE_TO_HUMAN' | 'UNCERTAIN' | 'OOD' | 'ERROR';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  label: string;
}

export interface CandidateRanking {
  label: string;
  score: number;
  prompt: string;
}

export interface RegionFinding {
  box: [number, number, number, number];
  area: number;
  spatial_anomaly_score: number;
  top_defect_label: string;
  top_similarity_score: number;
  confidence_margin: number;
  is_ambiguous: boolean;
  candidate_rankings: CandidateRanking[];
}

export interface ReliabilityIndicators {
  calibrated_threshold: number;
  conformal_p_value: number;
  target_false_alarm_rate: number;
  is_ood: boolean;
  ood_distance: number;
  ood_threshold: number;
  in_uncertainty_band: boolean;
  operational_state?: string;
  disposition?: string;
  guarantee_status?: string;
  human_review_recommended: boolean;
  review_reason: string;
}

export interface InspectionResponse {
  inspection_id: string;
  timestamp: string;
  category: string;
  status: InspectionStatus;
  image_anomaly_score: number;
  is_anomaly: boolean;
  threshold: number;
  num_defects_found: number;
  bounding_boxes: BoundingBox[];
  region_findings: RegionFinding[];
  textual_interpretation: string;
  reliability: ReliabilityIndicators;
  heatmap_base64?: string | null;
  mask_base64?: string | null;
}

export interface BaselineMetric {
  method: string;
  category: string;
  image_auroc: number;
  pixel_auroc: number;
  pixel_aupro: number;
  fps: number;
  supervision: string;
}

export interface AblationVariant {
  name: string;
  image_auroc: number;
  pixel_auroc: number;
  pixel_aupro: number;
  fps: number;
  description: string;
}

export interface FewShotPoint {
  k: number;
  image_auroc: number;
  pixel_auroc: number;
  pixel_aupro: number;
}

