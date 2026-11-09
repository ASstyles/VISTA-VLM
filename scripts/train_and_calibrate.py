#!/usr/bin/env python3
"""
VISTA-VLM Training, Calibration, and Validation Script (Phases 8, 11).
Trains VISTA on normal images with synthetic anomaly generation.
Calibrates non-conformity threshold on independent normal validation images using
Split Conformal Prediction.
Fits OOD prototype boundary on normal features.
Saves versioned model checkpoint and calibration parameters.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from vista_vlm.config.schema import VISTAConfig
from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.synthetic import SyntheticAnomalyGenerator
from vista_vlm.data.transforms import denormalize_image, VISTATransform
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.calibration.conformal import ConformalCalibrator
from vista_vlm.calibration.ood import OODDetector
from vista_vlm.calibration.selective import SelectivePredictor
from vista_vlm.evaluation.benchmark_runner import evaluate_vista_model
from vista_vlm.evaluation.visualizer import plot_roc_pr_curves, plot_calibration_reliability_diagram
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.train")

def train_one_epoch(
    model: VISTAAnomalyModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    synthetic_gen: SyntheticAnomalyGenerator,
    transform: VISTATransform,
    device: str = "cpu"
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images_norm = batch["image"] # [B, 3, H, W]
        B, C, H, W = images_norm.shape

        # Generate synthetic anomalies on the fly from normal training images
        augmented_images = []
        target_masks = []
        target_labels = []

        for b in range(B):
            img_uint8 = denormalize_image(images_norm[b]) # [H, W, 3] uint8
            # Pick source texture from another image in the batch if available
            src_uint8 = denormalize_image(images_norm[(b + 1) % B]) if B > 1 else None

            pert_img, gt_mask, is_anom = synthetic_gen(img_uint8, source_image=src_uint8)
            img_tensor, mask_tensor = transform(pert_img, gt_mask)

            augmented_images.append(img_tensor.unsqueeze(0))
            target_masks.append(mask_tensor.unsqueeze(0))
            target_labels.append(1.0 if is_anom else 0.0)

        batch_x = torch.cat(augmented_images, dim=0).to(device)
        batch_mask = torch.cat(target_masks, dim=0).to(device)
        batch_label = torch.tensor(target_labels, dtype=torch.float32, device=device)

        # Forward pass with deep supervision
        pred = model(batch_x, return_aux=True)
        loss_dict = model.compute_loss(pred, batch_mask, batch_label)

        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="VISTA-VLM Training & Calibration")
    parser.add_argument("--category", default="bottle", help="Dataset category to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_root", default="data/processed/mvtec", help="Root directory for dataset")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--target_far", type=float, default=0.05, help="Target false alarm rate alpha")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting training on device: {device} | Category: {args.category} | Epochs: {args.epochs}")

    # 1. Datasets & Loaders
    train_ds = MVTecDataset(args.data_root, category=args.category, split="train", image_size=518)
    val_cal_ds = MVTecDataset(args.data_root, category=args.category, split="val_calibration", image_size=518)
    test_ds = MVTecDataset(args.data_root, category=args.category, split="test", image_size=518)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_cal_loader = DataLoader(val_cal_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded {len(train_ds)} train normal samples, {len(val_cal_ds)} val_calibration normal samples, {len(test_ds)} test samples.")

    # 2. Model & Optimizer
    model = VISTAAnomalyModel(
        backbone_name="dinov2_vits14",
        embed_dim=384,
        out_indices=[4, 8, 12],
        proj_dim=256,
        image_size=518,
        fusion_type="global_local",
        use_film=True,
        freeze_backbone=True
    ).to(device)

    # Trainable parameters (fusion, decoder, global_head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters count: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    synthetic_gen = SyntheticAnomalyGenerator(anomaly_prob=0.5, seed=42)
    transform = VISTATransform(image_size=518, is_training=True)

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_val = train_one_epoch(model, train_loader, optimizer, synthetic_gen, transform, device=device)
        logger.info(f"Epoch [{epoch}/{args.epochs}] — Train Loss: {loss_val:.4f}")

    training_time = round(time.time() - t_start, 2)
    logger.info(f"Training completed in {training_time}s.")

    # 4. Save Model Checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"vista_mvtec_{args.category}.pth")
    model.save_checkpoint(
        ckpt_path,
        optimizer=optimizer,
        epoch=args.epochs,
        extra_metadata={
            "category": args.category,
            "epochs": args.epochs,
            "training_time_s": training_time,
            "train_samples": len(train_ds),
            "architecture": "v3.0.0-fpn-deep-supervision"
        }
    )

    # 5. Split Conformal Calibration on Holdout Normal Data
    logger.info("Executing Split Conformal Calibration on independent normal validation samples...")
    model.eval()
    cal_scores = []
    cal_features = []

    with torch.no_grad():
        for batch in val_cal_loader:
            images = batch["image"].to(device)
            outputs = model(images)
            scores = outputs["image_score"].cpu().numpy().tolist()
            feats = outputs["global_features"].cpu().numpy()
            cal_scores.extend(scores)
            cal_features.append(feats)

    cal_features_np = np.concatenate(cal_features, axis=0)

    # Fit Conformal Calibrator
    calibrator = ConformalCalibrator(target_false_alarm_rate=args.target_far)
    calibrated_threshold = calibrator.fit(cal_scores, category=args.category)
    calib_file = os.path.join(args.output_dir, f"calibration_mvtec.json")
    calibrator.save(calib_file)
    logger.info(f"Split Conformal Calibration: Category '{args.category}' threshold set to {calibrated_threshold:.4f} (alpha={args.target_far})")

    # Fit OOD Detector with regularized shrinkage covariance
    ood_detector = OODDetector(metric="mahalanobis", threshold_percentile=98.0)
    ood_detector.fit(cal_features_np, category=args.category)
    ood_file = os.path.join(args.output_dir, f"ood_mvtec.json")
    ood_detector.save(ood_file)
    logger.info(f"OOD Detector fitted on {len(cal_features_np)} normal vectors. Saved to {ood_file}")

    # Calibrate Selective Predictor Uncertainty Band
    sel_pred = SelectivePredictor()
    val_labels = [0] * len(cal_scores)
    opt_delta = sel_pred.calibrate_uncertainty_band(cal_scores, val_labels, calibrated_threshold=calibrated_threshold, target_risk=0.01)
    logger.info(f"Selective Predictor: Calibrated uncertainty band delta={opt_delta:.4f} ensuring <=1% selective risk")

    # 6. Evaluation on Test Benchmark Split
    logger.info("Evaluating VISTA-VLM on Test Benchmark Split...")
    eval_results = evaluate_vista_model(model, test_loader, device=device, calibrated_threshold=calibrated_threshold)
    
    logger.info(f"Test Results for VISTA-VLM ({args.category}):")
    logger.info(f"  Image AUROC: {eval_results['image_level']['auroc']:.4f} | AP: {eval_results['image_level']['average_precision']:.4f} | F1: {eval_results['image_level']['f1']:.4f}")
    logger.info(f"  Pixel AUROC: {eval_results['pixel_level']['pixel_auroc']:.4f} | Pixel AP: {eval_results['pixel_level']['pixel_ap']:.4f} | AUPRO: {eval_results['pixel_level']['aupro']:.4f}")
    logger.info(f"  Reliability: ECE: {eval_results['reliability']['ece']:.4f} | Empirical FAR: {eval_results['reliability']['empirical_false_alarm_rate']:.4f}")
    logger.info(f"  P50 Latency: {eval_results['efficiency']['p50_latency_ms']} ms | Throughput: {eval_results['efficiency']['throughput_fps']} FPS")

    # Generate and save diagnostic curves
    os.makedirs("benchmarks/results", exist_ok=True)
    plot_path = os.path.join("benchmarks", "results", f"roc_pr_{args.category}.png")
    plot_roc_pr_curves(
        y_true=np.array(eval_results["raw_scores"]["y_true"]),
        y_scores=np.array(eval_results["raw_scores"]["y_scores"]),
        save_path=plot_path,
        title_prefix=f"VISTA-VLM ({args.category})"
    )
    rel_plot_path = os.path.join("benchmarks", "results", f"calibration_{args.category}.png")
    plot_calibration_reliability_diagram(
        y_true=np.array(eval_results["raw_scores"]["y_true"]),
        y_probs=np.array(eval_results["raw_scores"]["y_scores"]),
        save_path=rel_plot_path
    )

    # Save detailed evaluation JSON
    eval_json_path = os.path.join("benchmarks", "results", f"vista_results_{args.category}.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation report and plots saved to benchmarks/results/")

if __name__ == "__main__":
    main()
