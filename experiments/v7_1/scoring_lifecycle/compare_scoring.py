#!/usr/bin/env python3
"""
Controlled Experiment: Training & Inference Scoring Formulations and Training Lifecycles (V7.1).
Compares:
  - Approach A: Memory-First Training (Fit memory bank -> train fusion/decoder/head with memory guide)
  - Approach B: Staged Optimization (Train decoder/head first -> fit memory bank -> fine-tune with memory guide)
  - Scoring Ablations:
      1. Global Head only
      2. Memory score only
      3. Spatial score only
      4. Global + Spatial fusion
      5. Memory + Spatial fusion
      6. Full Three-Component Fusion (Hybrid)
Evaluates on held-out validation partitions (val_tuning) to select optimal configuration without test leakage.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.synthetic import SyntheticAnomalyGenerator
from vista_vlm.data.transforms import denormalize_image, VISTATransform
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.metrics import compute_image_metrics, compute_pixel_metrics


def train_step(model, dataloader, optimizer, synthetic_gen, transform, device="cpu"):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        images_norm = batch["image"]
        B, C, H, W = images_norm.shape
        augmented_images, target_masks, target_labels = [], [], []
        for b in range(B):
            img_uint8 = denormalize_image(images_norm[b])
            src_uint8 = denormalize_image(images_norm[(b + 1) % B]) if B > 1 else None
            pert_img, gt_mask, is_anom = synthetic_gen(img_uint8, source_image=src_uint8)
            img_tensor, mask_tensor = transform(pert_img, gt_mask)
            augmented_images.append(img_tensor.unsqueeze(0))
            target_masks.append(mask_tensor.unsqueeze(0))
            target_labels.append(1.0 if is_anom else 0.0)

        batch_x = torch.cat(augmented_images, dim=0).to(device)
        batch_mask = torch.cat(target_masks, dim=0).to(device)
        batch_label = torch.tensor(target_labels, dtype=torch.float32, device=device)

        pred = model(batch_x, return_aux=True)
        loss_dict = model.compute_loss(pred, batch_mask, batch_label)
        loss = loss_dict["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches)


def evaluate_on_dataset(model, dataloader, scoring_mode="hybrid", device="cpu"):
    model.eval()
    orig_mode = model.scoring_mode
    model.scoring_mode = scoring_mode
    y_true, y_scores = [], []
    y_masks, y_maps = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            masks = batch["mask"].squeeze(1).cpu().numpy()

            out = model(imgs)
            scores = out["image_score"].cpu().numpy()
            maps = out["anomaly_map"].squeeze(1).cpu().numpy()

            y_true.extend(labels.tolist())
            y_scores.extend(scores.tolist())
            y_masks.append(masks)
            y_maps.append(maps)

    model.scoring_mode = orig_mode
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    y_masks_np = np.concatenate(y_masks, axis=0)
    y_maps_np = np.concatenate(y_maps, axis=0)

    img_m = compute_image_metrics(y_true_np, y_scores_np)
    pix_m = compute_pixel_metrics(y_masks_np, y_maps_np)
    return {
        "image_auroc": img_m["auroc"],
        "image_ap": img_m["average_precision"],
        "pixel_auroc": pix_m["pixel_auroc"],
        "iou": pix_m["iou"],
        "dice": pix_m["dice"]
    }


def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    categories = ["bottle", "cable", "capsule", "leather", "tile"]
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "categories": categories,
        "approach_comparison": {},
        "scoring_ablations": {}
    }

    synthetic_gen = SyntheticAnomalyGenerator(anomaly_prob=0.5, seed=42)
    transform = VISTATransform(image_size=518, is_training=True)

    print("=== STARTING CONTROLLED SCORING EXPERIMENTS ===")

    for cat in categories:
        print(f"\n--- Category: {cat} ---")
        train_ds = MVTecDataset("data/synthetic_fixtures/mvtec", category=cat, split="train")
        val_tune_ds = MVTecDataset("data/synthetic_fixtures/mvtec", category=cat, split="val_tuning")
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        tune_loader = DataLoader(val_tune_ds, batch_size=4, shuffle=False)

        # ---------------------------------------------------------------------
        # Approach A: Memory-First Training
        # 1. Fit memory bank first
        # 2. Train decoder & head with active memory guide
        # ---------------------------------------------------------------------
        print("Training Approach A (Memory-First)...")
        model_a = VISTAAnomalyModel(category=cat).to(device)
        model_a.fit_memory_bank(train_loader, device=device, max_patches=1000)
        opt_a = torch.optim.AdamW([p for p in model_a.parameters() if p.requires_grad], lr=1e-4)

        loss_a_history = []
        for ep in range(2):
            loss = train_step(model_a, train_loader, opt_a, synthetic_gen, transform, device=device)
            loss_a_history.append(loss)

        eval_a = evaluate_on_dataset(model_a, tune_loader, scoring_mode="hybrid", device=device)
        print(f"Approach A: Val Img AUROC={eval_a['image_auroc']:.4f}, Pix AUROC={eval_a['pixel_auroc']:.4f}, IoU={eval_a['iou']:.4f}")

        # ---------------------------------------------------------------------
        # Approach B: Staged Optimization
        # 1. Train decoder & head for 1 epoch with unfitted memory
        # 2. Fit memory bank
        # 3. Fine-tune for 1 epoch with active memory guide
        # ---------------------------------------------------------------------
        print("Training Approach B (Staged Optimization)...")
        model_b = VISTAAnomalyModel(category=cat).to(device)
        opt_b = torch.optim.AdamW([p for p in model_b.parameters() if p.requires_grad], lr=1e-4)

        loss_b_history = []
        loss1 = train_step(model_b, train_loader, opt_b, synthetic_gen, transform, device=device)
        loss_b_history.append(loss1)
        model_b.fit_memory_bank(train_loader, device=device, max_patches=1000)
        loss2 = train_step(model_b, train_loader, opt_b, synthetic_gen, transform, device=device)
        loss_b_history.append(loss2)

        eval_b = evaluate_on_dataset(model_b, tune_loader, scoring_mode="hybrid", device=device)
        print(f"Approach B: Val Img AUROC={eval_b['image_auroc']:.4f}, Pix AUROC={eval_b['pixel_auroc']:.4f}, IoU={eval_b['iou']:.4f}")

        results["approach_comparison"][cat] = {
            "approach_a": {"metrics": eval_a, "loss_history": loss_a_history},
            "approach_b": {"metrics": eval_b, "loss_history": loss_b_history}
        }

        # ---------------------------------------------------------------------
        # Scoring Ablations on Model A
        # ---------------------------------------------------------------------
        print("Running Scoring Ablations on Model A...")
        ablations = {}
        for mode in ["head_only", "memory_only", "spatial_only", "global_spatial", "memory_spatial", "hybrid"]:
            eval_mode = evaluate_on_dataset(model_a, tune_loader, scoring_mode=mode, device=device)
            ablations[mode] = eval_mode
            print(f"  Mode '{mode}': Img AUROC={eval_mode['image_auroc']:.4f}, Pix AUROC={eval_mode['pixel_auroc']:.4f}")

        results["scoring_ablations"][cat] = ablations

    out_json = "experiments/v7_1/scoring_lifecycle/results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment complete. Saved results to {out_json}")


if __name__ == "__main__":
    run_experiment()

