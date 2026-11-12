import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import precision_recall_curve

from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.metrics import compute_image_metrics, compute_pixel_metrics, compute_reliability_metrics
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.benchmark_runner")


def tune_vista_pixel_threshold(
    model: VISTAAnomalyModel,
    dataloader: DataLoader,
    device: str = "cpu"
) -> float:
    """
    Calibrate dense localization threshold for VISTA on independent validation data (val_tuning),
    preventing test-set label snooping.
    """
    if hasattr(model, "eval"):
        model.eval()
    y_masks = []
    y_pred_maps = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].squeeze(1).cpu().numpy()
            outputs = model(images)
            maps = outputs["anomaly_map"].squeeze(1).cpu().numpy()
            y_masks.append(masks)
            y_pred_maps.append(maps)

    y_t_eval = (np.concatenate(y_masks, axis=0).reshape(-1) > 0.5).astype(int)
    y_p_eval = np.concatenate(y_pred_maps, axis=0).reshape(-1)

    if len(y_t_eval) > 500_000:
        sub_idx = np.random.choice(len(y_t_eval), size=500_000, replace=False)
        y_t_eval = y_t_eval[sub_idx]
        y_p_eval = y_p_eval[sub_idx]

    precisions, recalls, thresholds = precision_recall_curve(y_t_eval, y_p_eval)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(y_p_eval))


def calibrate_baseline_thresholds(
    model: Any,
    dataloader: DataLoader,
    model_name: str,
    device: str = "cpu",
    category: str = "bottle"
) -> Tuple[float, float]:
    """
    Calibrates baseline image and pixel operating thresholds on independent validation data
    (e.g. val_tuning), ensuring strict parity with VISTA and eliminating test-label snooping.
    """
    if hasattr(model, "eval"):
        model.eval()
    y_true_list = []
    y_score_list = []
    y_mask_list = []
    y_pred_map_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            masks = batch["mask"].squeeze(1).cpu().numpy()

            if model_name in ("winclip", "anomalyclip"):
                outputs = model.predict(images, category=category)
            else:
                outputs = model.predict(images)

            scores = outputs["image_score"].cpu().numpy()
            maps = outputs["anomaly_map"].squeeze(1).cpu().numpy()

            y_true_list.extend(labels.tolist())
            y_score_list.extend(scores.tolist())
            y_mask_list.append(masks)
            y_pred_map_list.append(maps)

    y_true = np.array(y_true_list)
    y_scores = np.array(y_score_list)
    y_masks = np.concatenate(y_mask_list, axis=0)
    y_pred_maps = np.concatenate(y_pred_map_list, axis=0)

    # 1. Image threshold tuning
    if len(np.unique(y_true)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        img_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(y_scores))
    else:
        img_thresh = float(np.median(y_scores))

    # 2. Pixel threshold tuning
    y_t_eval = (y_masks.reshape(-1) > 0.5).astype(int)
    y_p_eval = y_pred_maps.reshape(-1)
    if len(y_t_eval) > 500_000:
        sub_idx = np.random.choice(len(y_t_eval), size=500_000, replace=False)
        y_t_eval = y_t_eval[sub_idx]
        y_p_eval = y_p_eval[sub_idx]

    if len(np.unique(y_t_eval)) > 1:
        prec_p, rec_p, thresh_p = precision_recall_curve(y_t_eval, y_p_eval)
        f1_p = 2 * (prec_p * rec_p) / (prec_p + rec_p + 1e-8)
        best_p_idx = np.argmax(f1_p)
        pix_thresh = float(thresh_p[best_p_idx]) if best_p_idx < len(thresh_p) else float(np.median(y_p_eval))
    else:
        pix_thresh = float(np.median(y_p_eval))

    return img_thresh, pix_thresh

def evaluate_vista_model(
    model: VISTAAnomalyModel,
    dataloader: DataLoader,
    device: str = "cpu",
    calibrated_threshold: float = 0.5,
    pixel_threshold: Optional[float] = None
) -> Dict[str, Any]:
    if hasattr(model, "eval"):
        model.eval()
    y_true_list = []
    y_score_list = []
    y_mask_list = []
    y_pred_map_list = []
    latencies = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            masks = batch["mask"].squeeze(1).cpu().numpy() # [B, H, W]

            t0 = time.time()
            outputs = model(images)
            lat = (time.time() - t0) * 1000.0 / images.shape[0]
            latencies.append(lat)

            scores = outputs["image_score"].cpu().numpy()
            maps = outputs["anomaly_map"].squeeze(1).cpu().numpy() # [B, H, W]

            y_true_list.extend(labels.tolist())
            y_score_list.extend(scores.tolist())
            y_mask_list.append(masks)
            y_pred_map_list.append(maps)

    y_true = np.array(y_true_list)
    y_scores = np.array(y_score_list)
    y_masks = np.concatenate(y_mask_list, axis=0)
    y_pred_maps = np.concatenate(y_pred_map_list, axis=0)

    img_metrics = compute_image_metrics(y_true, y_scores, threshold=calibrated_threshold)
    pix_metrics = compute_pixel_metrics(y_masks, y_pred_maps, threshold=pixel_threshold)
    rel_metrics = compute_reliability_metrics(y_true, y_scores, calibrated_threshold=calibrated_threshold)

    return {
        "image_level": img_metrics,
        "pixel_level": pix_metrics,
        "reliability": rel_metrics,
        "efficiency": {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "throughput_fps": round(1000.0 / float(np.mean(latencies)), 2)
        },
        "raw_scores": {
            "y_true": y_true.tolist(),
            "y_scores": y_scores.tolist()
        }
    }


def evaluate_baseline_model(
    model: Any,
    dataloader: DataLoader,
    model_name: str,
    device: str = "cpu",
    category: str = "bottle",
    calibrated_threshold: Optional[float] = None,
    pixel_threshold: Optional[float] = None
) -> Dict[str, Any]:
    if hasattr(model, "eval"):
        model.eval()
    y_true_list = []
    y_score_list = []
    y_mask_list = []
    y_pred_map_list = []
    latencies = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            masks = batch["mask"].squeeze(1).cpu().numpy()

            t0 = time.time()
            if model_name in ("winclip", "anomalyclip"):
                outputs = model.predict(images, category=category)
            else:
                outputs = model.predict(images)
            lat = (time.time() - t0) * 1000.0 / images.shape[0]
            latencies.append(lat)

            scores = outputs["image_score"].cpu().numpy()
            maps = outputs["anomaly_map"].squeeze(1).cpu().numpy()

            y_true_list.extend(labels.tolist())
            y_score_list.extend(scores.tolist())
            y_mask_list.append(masks)
            y_pred_map_list.append(maps)

    y_true = np.array(y_true_list)
    y_scores = np.array(y_score_list)
    y_masks = np.concatenate(y_mask_list, axis=0)
    y_pred_maps = np.concatenate(y_pred_map_list, axis=0)

    # Use independent holdout-calibrated threshold when available; NEVER optimize on test labels
    img_metrics = compute_image_metrics(y_true, y_scores, threshold=calibrated_threshold)
    pix_metrics = compute_pixel_metrics(y_masks, y_pred_maps, threshold=pixel_threshold)
    effective_thresh = calibrated_threshold if calibrated_threshold is not None else img_metrics["threshold"]
    rel_metrics = compute_reliability_metrics(y_true, y_scores, calibrated_threshold=effective_thresh)

    return {
        "image_level": img_metrics,
        "pixel_level": pix_metrics,
        "reliability": rel_metrics,
        "efficiency": {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "throughput_fps": round(1000.0 / float(np.mean(latencies)), 2)
        },
        "raw_scores": {
            "y_true": y_true.tolist(),
            "y_scores": y_scores.tolist()
        }
    }
