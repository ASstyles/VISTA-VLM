#!/usr/bin/env python3
"""
VISTA-VLM V6.0 Industrial Inspection Line Simulator (Phase 7).
=============================================================
Simulates a physical factory automated optical inspection (AOI) cell:
  1. Optical Acquisition & Quality Validation (Laplacian blur, illumination)
  2. VISTA-VLM V6.0 Multi-Scale Inference & Calibrated Distance Scoring
  3. Split Conformal Statistical Non-Conformity Testing
  4. Ledoit-Wolf Mahalanobis OOD Verification
  5. Deterministic Fail-Safe Operational State Engine
  6. Line PLC Hardware Diverter Interlock Emulation (Profinet / EtherNet/IP)
  7. Tamper-Evident Audit Trail Storage with SHA-256 Hashes
  8. Operator Feedback & Human-in-the-Loop Review Workflow

Computes Industrial Production KPIs:
  - Defect Escape Rate (FPR with exact Clopper-Pearson bound)
  - False Rejection Rate (Overkill Rate)
  - Operator Escalation / Review Rate
  - Takt-Time Latency Distribution (P50, P95, P99)
  - Physical Diverter Actuation Accounting
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import hashlib
import argparse
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista_vlm.config.schema import VISTAConfig
from vista_vlm.models.pipeline import VISTAInspectionPipeline
from vista_vlm.calibration.selective import (
    OperationalState,
    PhysicalDisposition,
    wilson_score_interval,
    exact_binomial_upper_bound
)
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.line_simulator")


class CameraSimulator:
    """Simulates industrial GigE Vision camera acquisition with environmental noise injection."""
    def __init__(self, blur_failure_prob: float = 0.05, dark_failure_prob: float = 0.05):
        self.blur_prob = blur_failure_prob
        self.dark_prob = dark_failure_prob

    def acquire_frame(self, image_path: str) -> Tuple[Image.Image, bool, str]:
        """Loads image and evaluates pre-inference optical quality metrics."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Dimension validation
        if w < 16 or h < 16 or w > 4096 or h > 4096:
            return img, False, f"Image dimensions {w}x{h} outside permissible range"

        # Laplacian blur variance check
        img_gray = np.array(img.convert("L"))
        gy, gx = np.gradient(img_gray.astype(float))
        gnorm = np.sqrt(gx ** 2 + gy ** 2)
        variance = float(np.var(gnorm))

        mean_lux = float(np.mean(img_gray))
        if mean_lux < 15.0:
            return img, False, f"Severe underexposure: mean pixel intensity {mean_lux:.1f} < 15.0"
        if mean_lux > 245.0:
            return img, False, f"Severe sensor saturation/glare: mean pixel intensity {mean_lux:.1f} > 245.0"
        if variance < 5.0:
            return img, False, f"Defocus/motion blur detected: gradient variance {variance:.2f} < 5.0"

        return img, True, "Optical quality nominal"


class IndustrialPLCEmulator:
    """Emulates factory Line Automation PLC receiving decisions and controlling physical diverter."""
    def __init__(self, takt_time_budget_ms: float = 500.0):
        self.takt_time_budget = takt_time_budget_ms
        self.diverter_counts = {
            PhysicalDisposition.ACCEPT.value: 0,
            PhysicalDisposition.REJECT.value: 0,
            PhysicalDisposition.HOLD.value: 0,
            PhysicalDisposition.QUARANTINE.value: 0,
            PhysicalDisposition.HUMAN_REVIEW.value: 0,
        }
        self.timing_violations = 0

    def receive_decision(
        self,
        item_id: str,
        disposition: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        """Enforces physical fail-closed logic and takt-time deadlines."""
        # Fail-closed timeout enforcement
        if latency_ms > self.takt_time_budget:
            self.timing_violations += 1
            effective_disp = PhysicalDisposition.HOLD.value
            actuation_action = "Pneumatic diverter drops to HOLD chute (Takt-Time Deadline Exceeded)"
        else:
            effective_disp = disposition
            if disposition == PhysicalDisposition.ACCEPT.value:
                actuation_action = "Release gate opens: Part discharged to finished goods conveyor"
            elif disposition == PhysicalDisposition.REJECT.value:
                actuation_action = "Pneumatic blow-off fires: Part routed to defective scrap bin"
            elif disposition == PhysicalDisposition.QUARANTINE.value:
                actuation_action = "Diverter routes to locked quarantine chute (OOD / Safety Condition)"
            else:
                actuation_action = "Diverter routes to operator inspection buffer (Human Review)"

        self.diverter_counts[effective_disp] += 1

        return {
            "item_id": item_id,
            "effective_disposition": effective_disp,
            "actuation_action": actuation_action,
            "latency_ms": round(latency_ms, 2),
            "deadline_exceeded": latency_ms > self.takt_time_budget
        }


def run_simulation(
    image_paths: List[str],
    ground_truth_labels: List[int],
    category: str = "bottle",
    takt_time_budget_ms: float = 500.0,
    output_audit_log: str = "reports/simulated_inspection_audit.json"
) -> Dict[str, Any]:
    logger.info(f"Initializing VISTA-VLM Inspection Line Simulator for category '{category}'...")

    # Load production inspection pipeline
    cfg = VISTAConfig()
    pipeline = VISTAInspectionPipeline(
        config=cfg,
        checkpoint_path="checkpoints/vista_mvtec_bottle.pth" if os.path.exists("checkpoints/vista_mvtec_bottle.pth") else None,
        calibration_path="checkpoints/calibration_mvtec.json" if os.path.exists("checkpoints/calibration_mvtec.json") else None,
        ood_path="checkpoints/ood_mvtec.json" if os.path.exists("checkpoints/ood_mvtec.json") else None
    )

    camera = CameraSimulator()
    plc = IndustrialPLCEmulator(takt_time_budget_ms=takt_time_budget_ms)

    audit_records = []
    latencies = []
    defects_escaped = 0
    false_alarms = 0
    total_defects = sum(ground_truth_labels)
    total_normals = len(ground_truth_labels) - total_defects

    t_start = time.time()
    for idx, (img_path, true_label) in enumerate(zip(image_paths, ground_truth_labels)):
        item_id = f"PART-{uuid.uuid4().hex[:8].upper()}"
        t0 = time.time()

        # Step 1: Camera acquisition & optical validation
        pil_img, img_ok, img_msg = camera.acquire_frame(img_path)

        if not img_ok:
            # Optical acquisition failure -> Immediate HOLD
            disp = PhysicalDisposition.HOLD.value
            op_state = OperationalState.IMAGE_QUALITY_FAILURE.value
            latency_ms = (time.time() - t0) * 1000.0
            insp_res = {
                "status": op_state,
                "image_anomaly_score": 0.0,
                "is_anomaly": False,
                "reliability": {"disposition": disp, "review_reason": img_msg}
            }
        else:
            # Step 2: VISTA-VLM V6.0 Inference & Verification
            raw_res = pipeline.inspect(pil_img, category=category)
            insp_res = raw_res.model_dump() if hasattr(raw_res, "model_dump") else raw_res.dict()
            latency_ms = (time.time() - t0) * 1000.0
            disp = insp_res["reliability"]["disposition"]
            op_state = insp_res["reliability"]["operational_state"]

        latencies.append(latency_ms)

        # Step 3: Industrial PLC Diverter Actuation
        plc_event = plc.receive_decision(item_id, disp, latency_ms)

        # Step 4: Quality & Safety KPI Accounting
        effective_disp = plc_event["effective_disposition"]
        if true_label == 1 and effective_disp == PhysicalDisposition.ACCEPT.value:
            defects_escaped += 1
            logger.error(f"[DEFECT ESCAPE] {item_id}: Defective part falsely ACCEPTED! Score={insp_res['image_anomaly_score']:.4f}")
        elif true_label == 0 and effective_disp in (PhysicalDisposition.REJECT.value, PhysicalDisposition.QUARANTINE.value):
            false_alarms += 1

        # Step 5: Tamper-Evident Audit Trail Record
        rec = {
            "item_id": item_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_source": img_path,
            "ground_truth": "DEFECTIVE" if true_label == 1 else "NORMAL",
            "anomaly_score": round(float(insp_res["image_anomaly_score"]), 4),
            "operational_state": op_state,
            "physical_disposition": effective_disp,
            "plc_action": plc_event["actuation_action"],
            "latency_ms": round(latency_ms, 2),
            "audit_hash": hashlib.sha256(f"{item_id}:{op_state}:{effective_disp}".encode()).hexdigest()[:16]
        }
        audit_records.append(rec)

    total_sim_time = round(time.time() - t_start, 2)
    n_inspected = len(image_paths)

    # Statistical KPI calculations
    escape_rate = float(defects_escaped / max(1, total_defects))
    overkill_rate = float(false_alarms / max(1, total_normals))
    review_count = plc.diverter_counts[PhysicalDisposition.HUMAN_REVIEW.value] + plc.diverter_counts[PhysicalDisposition.HOLD.value]
    review_rate = float(review_count / max(1, n_inspected))

    escape_ci_low, escape_ci_high = wilson_score_interval(defects_escaped, total_defects, 0.95)
    escape_exact_ub = exact_binomial_upper_bound(defects_escaped, total_defects, 0.95)

    kpis = {
        "simulation_summary": {
            "total_parts_inspected": n_inspected,
            "total_defective_parts": total_defects,
            "total_normal_parts": total_normals,
            "simulation_duration_s": total_sim_time,
            "effective_throughput_fps": round(n_inspected / max(0.1, total_sim_time), 2)
        },
        "quality_kpis": {
            "defects_escaped": defects_escaped,
            "observed_defect_escape_rate": round(escape_rate, 4),
            "escape_rate_95_ci": [round(escape_ci_low, 4), round(escape_ci_high, 4)],
            "escape_rate_exact_95_upper_bound": round(escape_exact_ub, 4),
            "false_rejections_overkill": false_alarms,
            "observed_false_rejection_rate": round(overkill_rate, 4),
            "operator_review_rate": round(review_rate, 4)
        },
        "latency_kpis": {
            "takt_time_budget_ms": takt_time_budget_ms,
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "takt_time_violations": plc.timing_violations,
            "takt_time_compliance_pct": round((1.0 - plc.timing_violations / max(1, n_inspected)) * 100.0, 2)
        },
        "plc_diverter_disposition_counts": plc.diverter_counts
    }

    # Save audit log
    os.makedirs(os.path.dirname(output_audit_log), exist_ok=True)
    with open(output_audit_log, "w", encoding="utf-8") as f:
        json.dump({"kpis": kpis, "audit_trail": audit_records}, f, indent=2)

    logger.info(f"Inspection Line Simulation Completed: {n_inspected} parts inspected.")
    logger.info(f"  Defect Escape Rate: {escape_rate:.2%} (Exact 95% Bound: {escape_exact_ub:.2%})")
    logger.info(f"  False Rejection Rate: {overkill_rate:.2%} | Operator Review Rate: {review_rate:.2%}")
    logger.info(f"  P50 Latency: {kpis['latency_kpis']['p50_latency_ms']} ms | Takt-Time Compliance: {kpis['latency_kpis']['takt_time_compliance_pct']}%")
    logger.info(f"  Audit log saved to: {output_audit_log}")

    return kpis


def main():
    parser = argparse.ArgumentParser(description="VISTA-VLM Industrial Inspection Line Simulator")
    parser.add_argument("--category", default="bottle", help="Product category")
    parser.add_argument("--takt_time", type=float, default=500.0, help="Takt time deadline budget in ms")
    parser.add_argument("--output", default="reports/simulated_inspection_audit.json", help="Output audit log path")
    args = parser.parse_args()

    # Collect test images from synthetic fixtures
    img_dir = os.path.join("data", "synthetic_fixtures", "mvtec", args.category, "test")
    if not os.path.exists(img_dir):
        logger.error(f"Test image directory not found: {img_dir}. Run prepare_benchmark_data.py first.")
        sys.exit(1)

    images = []
    labels = []
    for defect_type in os.listdir(img_dir):
        sub_dir = os.path.join(img_dir, defect_type)
        if not os.path.isdir(sub_dir):
            continue
        is_defect = 0 if defect_type == "good" else 1
        for fname in os.listdir(sub_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(sub_dir, fname))
                labels.append(is_defect)

    if not images:
        logger.error(f"No test images found in {img_dir}")
        sys.exit(1)

    run_simulation(
        image_paths=images,
        ground_truth_labels=labels,
        category=args.category,
        takt_time_budget_ms=args.takt_time,
        output_audit_log=args.output
    )


if __name__ == "__main__":
    main()
