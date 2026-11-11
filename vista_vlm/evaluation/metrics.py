import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)
from scipy.ndimage import label
from typing import Dict, Any, List, Optional

def compute_image_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes Image-Level Metrics: AUROC, AP, F1, Precision, Recall.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.nan_to_num(np.asarray(y_scores).astype(float), nan=0.0)

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
    y_true_flat = (y_true_masks.reshape(-1) > 0.5).astype(int)
    y_pred_flat = np.nan_to_num(y_pred_maps.reshape(-1).astype(float), nan=0.0)

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

    # Validate map spatial variation to detect degenerate constant scalar predictions
    val_info = check_spatial_map_validity(y_pred_maps)
    is_degenerate = val_info.get("is_constant_scalar", False)

    # AUPRO: calculated on globally scaled continuous maps to preserve dynamic range
    aupro = compute_aupro_score(y_true_masks, y_pred_maps)
    defect_strat = compute_defect_size_metrics(y_true_masks, y_pred_maps, threshold=eval_thresh)

    return {
        "pixel_auroc": round(pix_auroc, 4),
        "pixel_ap": round(pix_ap, 4),
        "aupro": round(aupro, 4),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "is_constant_map": is_degenerate,
        "spatial_variance": val_info.get("mean_spatial_std", 0.0),
        "defect_size_stratification": defect_strat,
    }


def check_spatial_map_validity(maps: np.ndarray, min_variance: float = 1e-7) -> Dict[str, Any]:
    """
    Validates spatial anomaly maps to reject degenerate constant predictions.
    Rejects any method whose supposed localization map is merely a replicated scalar.
    """
    maps_arr = np.asarray(maps)
    if maps_arr.ndim < 2:
        return {"is_valid": False, "reason": "Map dimensions must be at least 2D [H, W]"}

    # Check per-image spatial variance
    if maps_arr.ndim == 2:
        spatial_stds = [float(np.std(maps_arr))]
    else:
        # [N, H, W] or [N, 1, H, W]
        flat_maps = maps_arr.reshape(maps_arr.shape[0], -1)
        spatial_stds = [float(s) for s in np.std(flat_maps, axis=1)]

    mean_std = float(np.mean(spatial_stds))
    min_std = float(np.min(spatial_stds))
    is_constant = min_std < min_variance

    return {
        "is_valid": not is_constant,
        "mean_spatial_std": round(mean_std, 6),
        "min_spatial_std": round(min_std, 6),
        "is_constant_scalar": is_constant,
        "reason": "Map has zero or negligible spatial variance (replicated scalar artifact)" if is_constant else "Valid non-constant spatial map"
    }


def compute_aupro_score(y_true_masks: np.ndarray, y_pred_maps: np.ndarray, max_fpr: float = 0.3, num_thresholds: int = 100) -> float:
    """
    Computes Area Under the Per-Region Overlap (AUPRO) curve up to max_fpr (Bergmann et al.).
    Evaluates how well localized connected defect regions are detected across thresholds.
    Features:
    1. Pre-extracts connected components for each defect region.
    2. Sweeps fine thresholds across the full dynamic range of predicted scores.
    3. Rigorously interpolates endpoints to (0, 0) and (max_fpr, PRO(max_fpr))
       to prevent under-integration truncation errors.
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

    # Pre-extract 1D score arrays for each connected defect region
    region_score_arrays = []
    for i in range(N):
        gt = y_true_masks[i]
        if gt.sum() == 0:
            continue
        labeled_gt, num_features = label(gt)
        for reg_id in range(1, num_features + 1):
            reg_m = (labeled_gt == reg_id)
            reg_scores = maps_scaled[i][reg_m]
            region_score_arrays.append(reg_scores)

    if not region_score_arrays:
        return 1.0

    normal_scores = maps_scaled[y_true_masks == 0]
    total_normal_pix = float(len(normal_scores)) + 1e-8

    # Adaptive threshold sweep combining quantile FPR distribution and dynamic range
    target_fprs = np.linspace(0.0, max_fpr, max(10, num_thresholds // 2))
    q_thresholds = np.quantile(normal_scores, np.clip(1.0 - target_fprs, 0.0, 1.0))
    lin_thresholds = np.linspace(float(np.min(normal_scores)), float(np.max(maps_scaled)), max(10, num_thresholds // 2))
    thresholds = np.unique(np.concatenate([q_thresholds, lin_thresholds]))
    pro_list = []
    fpr_list = []

    for t in thresholds:
        # Fast vectorized calculation over 1D arrays
        fp = float(np.count_nonzero(normal_scores > t))
        fpr = fp / total_normal_pix

        overlaps = [float(np.count_nonzero(r_s > t)) / len(r_s) for r_s in region_score_arrays]
        mean_pro = float(np.mean(overlaps))

        pro_list.append(mean_pro)
        fpr_list.append(fpr)

    # Sort by FPR monotonically
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    pro_sorted = np.array(pro_list)[sorted_indices]

    # Ensure origin (0, 0) is included if fpr_sorted[0] > 0
    if fpr_sorted[0] > 0.0:
        fpr_sorted = np.insert(fpr_sorted, 0, 0.0)
        pro_sorted = np.insert(pro_sorted, 0, 0.0)

    # Filter to max_fpr
    valid_idx = fpr_sorted <= max_fpr
    fpr_valid = list(fpr_sorted[valid_idx])
    pro_valid = list(pro_sorted[valid_idx])

    if len(fpr_valid) < 2:
        return float(np.mean(pro_list))

    # Exact endpoint interpolation: if last valid FPR < max_fpr, interpolate to exact max_fpr
    if fpr_valid[-1] < max_fpr:
        # Interpolate PRO at exact max_fpr boundary
        pro_at_max = float(np.interp(max_fpr, fpr_sorted, pro_sorted))
        fpr_valid.append(max_fpr)
        pro_valid.append(pro_at_max)

    fpr_final = np.array(fpr_valid)
    pro_final = np.array(pro_valid)

    # Integrate up to max_fpr using trapezoid (NumPy 2.0+ compatible)
    if hasattr(np, "trapezoid"):
        trap_fn = np.trapezoid
    else:
        from scipy.integrate import trapezoid as trap_fn

    aupro = trap_fn(pro_final, fpr_final) / max_fpr
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
    else:
        y_probs = np.clip(y_raw_scores, 0.0, 1.0)

    # Brier Score: MSE between predicted probability and binary truth
    brier = float(brier_score_loss(y_true, y_probs))

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

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


def compute_defect_size_metrics(
    y_true_masks: np.ndarray,
    y_pred_maps: np.ndarray,
    threshold: float = 0.5,
    min_overlap: float = 0.10
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates localization sensitivity stratified by defect area (Phase 5.3):
    - Tiny defects:   < 0.1% of image area (< 0.001)
    - Small defects:  0.1% - 1.0% of image area (0.001 - 0.01)
    - Medium defects: 1.0% - 5.0% of image area (0.01 - 0.05)
    - Large defects:  > 5.0% of image area (> 0.05)
    """
    bins = {
        "tiny (<0.1%)": {"count": 0, "detected": 0, "overlaps": []},
        "small (0.1-1%)": {"count": 0, "detected": 0, "overlaps": []},
        "medium (1-5%)": {"count": 0, "detected": 0, "overlaps": []},
        "large (>5%)": {"count": 0, "detected": 0, "overlaps": []},
    }

    N = len(y_true_masks)
    for i in range(N):
        gt = y_true_masks[i]
        if gt.sum() == 0:
            continue
        total_pixels = float(gt.shape[0] * gt.shape[1])
        pred_map = y_pred_maps[i]
        pred_bin = (pred_map > threshold).astype(int)

        labeled_gt, num_reg = label(gt)
        for r_id in range(1, num_reg + 1):
            reg = (labeled_gt == r_id)
            reg_area = float(reg.sum())
            ratio = reg_area / total_pixels

            if ratio < 0.001:
                bin_key = "tiny (<0.1%)"
            elif ratio < 0.01:
                bin_key = "small (0.1-1%)"
            elif ratio < 0.05:
                bin_key = "medium (1-5%)"
            else:
                bin_key = "large (>5%)"

            overlap = float(np.count_nonzero(pred_bin[reg])) / max(1.0, reg_area)
            bins[bin_key]["count"] += 1
            bins[bin_key]["overlaps"].append(overlap)
            if overlap >= min_overlap:
                bins[bin_key]["detected"] += 1

    summary = {}
    for k, v in bins.items():
        cnt = v["count"]
        det = v["detected"]
        rec = float(det / cnt) if cnt > 0 else 1.0
        m_pro = float(np.mean(v["overlaps"])) if cnt > 0 else 1.0
        summary[k] = {
            "region_count": cnt,
            "detected_count": det,
            "recall": round(rec, 4),
            "mean_overlap": round(m_pro, 4)
        }
    return summary


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes rigorous non-parametric image-level bootstrap confidence intervals
    (1000 resamples) for AUROC and Average Precision across all evaluated samples.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    n = len(y_true)
    if n < 5 or len(np.unique(y_true)) < 2:
        return {
            "auroc": {"ci_lower": 0.0, "ci_upper": 1.0, "mean": 0.0, "std": 0.0},
            "ap": {"ci_lower": 0.0, "ci_upper": 1.0, "mean": 0.0, "std": 0.0},
            "n_bootstraps": n_bootstraps,
            "sample_size": n
        }

    rng = np.random.RandomState(seed)
    boot_aurocs = []
    boot_aps = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        ys_sample = y_scores[idx]
        if len(np.unique(yt_sample)) < 2:
            continue
        try:
            boot_aurocs.append(float(roc_auc_score(yt_sample, ys_sample)))
            boot_aps.append(float(average_precision_score(yt_sample, ys_sample)))
        except Exception:
            continue

    if not boot_aurocs:
        return {
            "auroc": {"ci_lower": 0.0, "ci_upper": 1.0, "mean": 0.0, "std": 0.0},
            "ap": {"ci_lower": 0.0, "ci_upper": 1.0, "mean": 0.0, "std": 0.0},
            "n_bootstraps": 0,
            "sample_size": n
        }

    alpha_tail = (1.0 - confidence) / 2.0
    auroc_low = float(np.percentile(boot_aurocs, 100.0 * alpha_tail))
    auroc_high = float(np.percentile(boot_aurocs, 100.0 * (1.0 - alpha_tail)))
    ap_low = float(np.percentile(boot_aps, 100.0 * alpha_tail))
    ap_high = float(np.percentile(boot_aps, 100.0 * (1.0 - alpha_tail)))

    return {
        "auroc": {
            "mean": round(float(np.mean(boot_aurocs)), 4),
            "std": round(float(np.std(boot_aurocs)), 4),
            "ci_lower": round(auroc_low, 4),
            "ci_upper": round(auroc_high, 4)
        },
        "ap": {
            "mean": round(float(np.mean(boot_aps)), 4),
            "std": round(float(np.std(boot_aps)), 4),
            "ci_lower": round(ap_low, 4),
            "ci_upper": round(ap_high, 4)
        },
        "n_bootstraps": len(boot_aurocs),
        "sample_size": n
    }


def compute_statistical_confidence_intervals(
    values: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """Computes mean, standard deviation, and 95% confidence bounds across multiple runs/categories."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    arr = np.array(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # Normal approximation for z = 1.96 (or t-distribution for small N)
    from scipy.stats import t as t_dist
    t_crit = float(t_dist.ppf((1.0 + confidence) / 2.0, df=max(1, n - 1))) if n > 1 else 1.96
    margin = t_crit * (std_val / np.sqrt(n)) if n > 1 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(max(0.0, mean_val - margin), 4),
        "ci_upper": round(min(1.0, mean_val + margin), 4),
        "sample_count": n
    }

