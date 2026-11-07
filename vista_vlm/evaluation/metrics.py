import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix
)
from scipy.ndimage import label
from typing import Dict, Any, List, Tuple, Optional

def compute_image_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes Image-Level Metrics: AUROC, AP, F1, Precision, Recall.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    # If only one class present in test set, AUROC cannot be calculated; return 1.0 or fallback
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))
    else:
        auroc = 1.0
        ap = 1.0

    # Determine threshold (either calibrated or optimal F1)
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    else:
        best_thresh = float(threshold)

    y_pred = (y_scores > best_thresh).astype(int)
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "threshold": round(best_thresh, 4),
    }


def compute_pixel_metrics(
    y_true_masks: np.ndarray,
    y_pred_maps: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes Pixel-Level Metrics: Pixel AUROC, Pixel AP, IoU, Dice.
    Args:
        y_true_masks: [N, H, W] binary masks
        y_pred_maps: [N, H, W] continuous anomaly heatmaps in [0, 1]
    """
    y_true_flat = y_true_masks.reshape(-1).astype(int)
    y_pred_flat = y_pred_maps.reshape(-1).astype(float)

    # Subsample if pixel count is huge (e.g. > 2 million pixels) to ensure fast exact computation
    if len(y_true_flat) > 1_000_000:
        subsample_idx = np.random.choice(len(y_true_flat), size=1_000_000, replace=False)
        y_t_eval = y_true_flat[subsample_idx]
        y_p_eval = y_pred_flat[subsample_idx]
    else:
        y_t_eval = y_true_flat
        y_p_eval = y_pred_flat

    if len(np.unique(y_t_eval)) > 1:
        pix_auroc = float(roc_auc_score(y_t_eval, y_p_eval))
        pix_ap = float(average_precision_score(y_t_eval, y_p_eval))
    else:
        pix_auroc = 1.0
        pix_ap = 1.0

    # Determine evaluation threshold (calibrated or optimal F1)
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_t_eval, y_p_eval)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        eval_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(y_p_eval))
    else:
        eval_thresh = float(threshold)

    pred_bin = (y_pred_flat > eval_thresh).astype(int)

    intersection = np.sum(pred_bin * y_true_flat)
    union = np.sum(pred_bin) + np.sum(y_true_flat) - intersection
    iou = float(intersection / (union + 1e-8))
    dice = float((2.0 * intersection) / (np.sum(pred_bin) + np.sum(y_true_flat) + 1e-8))

    # AUPRO: calculated on globally scaled continuous maps to preserve dynamic range
    aupro = compute_aupro_score(y_true_masks, y_pred_maps)

    return {
        "pixel_auroc": round(pix_auroc, 4),
        "pixel_ap": round(pix_ap, 4),
        "aupro": round(aupro, 4),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
    }


def compute_aupro_score(y_true_masks: np.ndarray, y_pred_maps: np.ndarray, max_fpr: float = 0.3) -> float:
    """
    Computes Area Under the Per-Region Overlap (AUPRO) curve up to max_fpr.
    Evaluates how well localized connected defect regions are detected across thresholds.
    Globally normalizes maps to [0, 1] if unnormalized distances are passed.
    """
    N = len(y_true_masks)
    if N == 0:
        return 0.0

    # Ensure continuous scores are mapped to [0, 1] for threshold sweep
    min_v = float(np.min(y_pred_maps))
    max_v = float(np.max(y_pred_maps))
    if max_v > min_v:
        maps_scaled = (y_pred_maps - min_v) / (max_v - min_v)
    else:
        maps_scaled = y_pred_maps

    # Pre-extract connected components for each image once
    region_masks = []
    for i in range(N):
        gt = y_true_masks[i]
        if gt.sum() == 0:
            continue
        labeled_gt, num_features = label(gt)
        for reg_id in range(1, num_features + 1):
            reg_m = (labeled_gt == reg_id)
            region_masks.append((i, reg_m, reg_m.sum()))

    if not region_masks:
        return 1.0

    thresholds = np.linspace(0.0, 1.0, 50)
    pro_list = []
    fpr_list = []
    normal_pixels = (y_true_masks == 0)
    total_normal_pix = float(normal_pixels.sum()) + 1e-8

    for t in thresholds:
        pred_bin = (maps_scaled > t)
        
        # Calculate per-region overlap
        total_overlap = 0.0
        for img_idx, reg_m, reg_area in region_masks:
            overlap = (pred_bin[img_idx] & reg_m).sum()
            total_overlap += overlap / (reg_area + 1e-8)

        mean_pro = total_overlap / len(region_masks)

        # Calculate FPR across all normal pixels
        fp = (pred_bin & normal_pixels).sum()
        fpr = float(fp) / total_normal_pix

        pro_list.append(mean_pro)
        fpr_list.append(fpr)

    # Sort by FPR
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    pro_sorted = np.array(pro_list)[sorted_indices]

    # Integrate up to max_fpr using trapezoid (NumPy 2.0+ compatible)
    valid_idx = fpr_sorted <= max_fpr
    if valid_idx.sum() < 2:
        return float(np.mean(pro_list))

    # Use trapezoid
    if hasattr(np, "trapezoid"):
        trap_fn = np.trapezoid
    else:
        from scipy.integrate import trapezoid as trap_fn

    aupro = trap_fn(pro_sorted[valid_idx], fpr_sorted[valid_idx]) / max_fpr
    return float(np.clip(aupro, 0.0, 1.0))


def compute_reliability_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    calibrated_threshold: float,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Reliability & Calibration Metrics:
    - Expected Calibration Error (ECE)
    - Brier Score
    - Empirical False Alarm Rate on normal samples
    """
    y_true = np.asarray(y_true).astype(int)
    y_raw_scores = np.asarray(y_probs).astype(float)

    # Scale continuous distance scores into [0, 1] if unnormalized
    min_s = float(np.min(y_raw_scores))
    max_s = float(np.max(y_raw_scores))
    if max_s > min_s and (min_s < 0.0 or max_s > 1.0):
        y_probs = (y_raw_scores - min_s) / (max_s - min_s)
        cal_thresh = (calibrated_threshold - min_s) / (max_s - min_s)
    else:
        y_probs = np.clip(y_raw_scores, 0.0, 1.0)
        cal_thresh = calibrated_threshold

    # Brier Score: MSE between predicted probability and binary truth
    brier = float(brier_score_loss(y_true, y_probs))

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Empirical False Alarm Rate (FPR on normal images at calibrated threshold)
    normal_mask = (y_true == 0)
    if normal_mask.sum() > 0:
        false_alarms = np.sum((y_raw_scores[normal_mask] > calibrated_threshold))
        empirical_far = float(false_alarms / normal_mask.sum())
    else:
        empirical_far = 0.0

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier, 4),
        "empirical_false_alarm_rate": round(empirical_far, 4),
    }
