import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from typing import Dict, Any

from scripts.run_v7_qualification import get_v7_loaders, train_one_epoch_synthetic
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.metrics import compute_pixel_metrics
from vista_vlm.data.synthetic import SyntheticAnomalyGenerator
from vista_vlm.data.transforms import VISTATransform

def evaluate_localization_on_loader(
    model: VISTAAnomalyModel,
    loader,
    device: str = "cpu",
    pixel_threshold: float = 0.5
) -> Dict[str, Any]:
    model.eval()
    y_masks = []
    y_pred_maps = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].squeeze(1).cpu().numpy()
            out = model(imgs)
            maps = out["anomaly_map"].squeeze(1).cpu().numpy()
            y_masks.append(masks)
            y_pred_maps.append(maps)

    all_masks = np.concatenate(y_masks, axis=0)
    all_maps = np.concatenate(y_pred_maps, axis=0)
    return compute_pixel_metrics(all_masks, all_maps, threshold=pixel_threshold)

def tune_pixel_threshold(model: VISTAAnomalyModel, val_loader, device: str = "cpu") -> float:
    model.eval()
    y_masks = []
    y_pred_maps = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].squeeze(1).cpu().numpy()
            out = model(imgs)
            maps = out["anomaly_map"].squeeze(1).cpu().numpy()
            y_masks.append(masks)
            y_pred_maps.append(maps)

    all_masks = (np.concatenate(y_masks, axis=0).reshape(-1) > 0.5).astype(int)
    all_maps = np.concatenate(y_pred_maps, axis=0).reshape(-1)

    # Search threshold maximizing IoU on validation masks
    best_iou = -1.0
    best_thresh = 0.5
    for t in np.linspace(0.1, 0.9, 33):
        pred_bin = (all_maps > t).astype(int)
        inter = np.sum(pred_bin * all_masks)
        union = np.sum(pred_bin) + np.sum(all_masks) - inter
        iou = inter / (union + 1e-8)
        if iou > best_iou:
            best_iou = iou
            best_thresh = float(t)
    return best_thresh

def run_localization_ablations():
    categories = [
        {"dataset": "mvtec", "name": "bottle", "type": "rigid_object"},
        {"dataset": "mvtec", "name": "cable", "type": "structural_wiring"},
        {"dataset": "mvtec", "name": "capsule", "type": "small_component"},
        {"dataset": "mvtec", "name": "leather", "type": "texture_surface"},
        {"dataset": "mvtec", "name": "tile", "type": "planar_texture"}
    ]
    device = "cpu"
    synthetic_gen = SyntheticAnomalyGenerator(anomaly_prob=0.5, seed=42)
    transform = VISTATransform(image_size=518, is_training=True)

    results: Dict[str, Any] = {}
    weights_to_test = [0.65, 0.35, 0.15, 0.0]

    for cat in categories:
        cname = cat["name"]
        print(f"\n=== Running Localization Ablations for category: {cname} ===")
        loaders = get_v7_loaders(cat, image_size=518, batch_size=4)
        train_loader, _cal_loader, tune_loader, test_loader = loaders[1], loaders[2], loaders[3], loaders[4]

        # Train base model with Approach A (Memory-First)
        model = VISTAAnomalyModel(
            backbone_name="dinov2_vits14",
            embed_dim=384,
            out_indices=[4, 8, 12],
            proj_dim=256,
            image_size=518,
            memory_fusion_weight=0.65,
            category=cname
        ).to(device)

        # Fit memory bank
        model.fit_memory_bank(train_loader, device=device, max_patches=1000, category=cname)

        # Train 1 epoch
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        train_one_epoch_synthetic(model, train_loader, opt, synthetic_gen, transform, device=device)

        cat_results: Dict[str, Any] = {}
        for w in weights_to_test:
            model.memory_fusion_weight = w
            # Tune pixel threshold on tune_loader
            tuned_t = tune_pixel_threshold(model, tune_loader, device=device)
            metrics_tuned = evaluate_localization_on_loader(model, test_loader, device=device, pixel_threshold=tuned_t)
            metrics_default = evaluate_localization_on_loader(model, test_loader, device=device, pixel_threshold=0.5)

            cat_results[f"weight_{w}"] = {
                "tuned_threshold": tuned_t,
                "metrics_tuned": {
                    "pixel_auroc": metrics_tuned["pixel_auroc"],
                    "aupro": metrics_tuned["aupro"],
                    "iou": metrics_tuned["iou"],
                    "dice": metrics_tuned["dice"],
                    "defect_size_stratification": metrics_tuned["defect_size_stratification"]
                },
                "metrics_default_0.5": {
                    "pixel_auroc": metrics_default["pixel_auroc"],
                    "aupro": metrics_default["aupro"],
                    "iou": metrics_default["iou"],
                    "dice": metrics_default["dice"]
                }
            }
            print(f"Weight {w}: Tuned t={tuned_t:.3f} -> IoU={metrics_tuned['iou']:.4f}, Dice={metrics_tuned['dice']:.4f}, AUPRO={metrics_tuned['aupro']:.4f} | Default t=0.5 -> IoU={metrics_default['iou']:.4f}")

        results[cname] = cat_results

    os.makedirs("experiments/v7_1/localization", exist_ok=True)
    with open("experiments/v7_1/localization/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved localization ablation results to experiments/v7_1/localization/results.json")

if __name__ == "__main__":
    run_localization_ablations()
