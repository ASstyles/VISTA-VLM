import os
import sys
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from vista_vlm.config.schema import VISTAConfig
from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.models.baselines import PatchCore, EfficientAD, DRAEM, PaDiM, WinCLIP, AnomalyCLIP
from vista_vlm.evaluation.metrics import compute_image_metrics, compute_pixel_metrics, compute_reliability_metrics
from vista_vlm.evaluation.visualizer import plot_roc_pr_curves, plot_calibration_reliability_diagram
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.benchmark_runner")

def evaluate_vista_model(
    model: VISTAAnomalyModel,
    dataloader: DataLoader,
    device: str = "cpu",
    calibrated_threshold: float = 0.5
) -> Dict[str, Any]:
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
    pix_metrics = compute_pixel_metrics(y_masks, y_pred_maps, threshold=calibrated_threshold)
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
    category: str = "bottle"
) -> Dict[str, Any]:
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

    img_metrics = compute_image_metrics(y_true, y_scores)
    pix_metrics = compute_pixel_metrics(y_masks, y_pred_maps)
    rel_metrics = compute_reliability_metrics(y_true, y_scores, calibrated_threshold=img_metrics["threshold"])

    return {
        "image_level": img_metrics,
        "pixel_level": pix_metrics,
        "reliability": rel_metrics,
        "efficiency": {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "throughput_fps": round(1000.0 / float(np.mean(latencies)), 2)
        }
    }
