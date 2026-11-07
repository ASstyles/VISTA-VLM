#!/usr/bin/env python3
"""
VISTA-VLM Baselines Benchmark Script (Phases 3, 4).
Runs and evaluates:
1. PatchCore (Phase 3)
2. EfficientAD (Phase 3)
3. DRAEM (Phase 3)
4. PaDiM (Phase 3)
5. WinCLIP (Phase 4)
6. AnomalyCLIP (Phase 4)
Produces consolidated comparative benchmark metrics.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
from torch.utils.data import DataLoader

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.models.baselines import PatchCore, EfficientAD, DRAEM, PaDiM, WinCLIP, AnomalyCLIP
from vista_vlm.evaluation.benchmark_runner import evaluate_baseline_model
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.run_baselines")

def main():
    parser = argparse.ArgumentParser(description="VISTA-VLM Baselines Benchmark")
    parser.add_argument("--category", default="bottle", help="Dataset category")
    parser.add_argument("--data_root", default="data/processed/mvtec", help="Data root")
    parser.add_argument("--output_file", default="benchmarks/results/baselines_comparison.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Baselines Benchmark for category '{args.category}' on {device}...")

    # Datasets
    train_ds = MVTecDataset(args.data_root, category=args.category, split="train", image_size=518)
    test_ds = MVTecDataset(args.data_root, category=args.category, split="test", image_size=518)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    baseline_results = {}

    # 1. PatchCore
    logger.info("Evaluating PatchCore...")
    patchcore = PatchCore(embed_dim=384, coreset_sampling_ratio=0.10, image_size=518)
    patchcore.fit(train_loader, device=device)
    baseline_results["PatchCore"] = evaluate_baseline_model(patchcore, test_loader, "patchcore", device=device)

    # 2. PaDiM
    logger.info("Evaluating PaDiM...")
    padim = PaDiM(embed_dim=64, image_size=518)
    padim.fit(train_loader, device=device)
    baseline_results["PaDiM"] = evaluate_baseline_model(padim, test_loader, "padim", device=device)

    # 3. EfficientAD
    logger.info("Evaluating EfficientAD...")
    efficientad = EfficientAD(image_size=518, feature_dim=64).to(device)
    opt = torch.optim.Adam(list(efficientad.student.parameters()) + list(efficientad.autoencoder.parameters()), lr=1e-3)
    # Quick fit on normal training batches
    for batch in train_loader:
        efficientad.fit_step(batch["image"].to(device), opt)
    baseline_results["EfficientAD"] = evaluate_baseline_model(efficientad, test_loader, "efficientad", device=device)

    # 4. DRAEM
    logger.info("Evaluating DRAEM...")
    draem = DRAEM(image_size=518).to(device)
    baseline_results["DRAEM"] = evaluate_baseline_model(draem, test_loader, "draem", device=device)

    # 5. WinCLIP (Zero-shot VLM)
    logger.info("Evaluating WinCLIP...")
    winclip = WinCLIP(embed_dim=512, image_size=518).to(device)
    baseline_results["WinCLIP"] = evaluate_baseline_model(winclip, test_loader, "winclip", device=device, category=args.category)

    # 6. AnomalyCLIP (Zero-shot VLM)
    logger.info("Evaluating AnomalyCLIP...")
    anomclip = AnomalyCLIP(embed_dim=512, image_size=518).to(device)
    baseline_results["AnomalyCLIP"] = evaluate_baseline_model(anomclip, test_loader, "anomalyclip", device=device, category=args.category)

    # Load VISTA-VLM results if available to combine
    vista_res_path = os.path.join("benchmarks", "results", f"vista_results_{args.category}.json")
    if os.path.exists(vista_res_path):
        with open(vista_res_path, "r") as f:
            v_data = json.load(f)
            # Remove raw scores for clean table presentation
            v_clean = {k: v for k, v in v_data.items() if k != "raw_scores"}
            baseline_results["VISTA-VLM (Ours)"] = v_clean

    # Save summary
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(baseline_results, f, indent=2)

    logger.info("=" * 70)
    logger.info("CONSOLIDATED BENCHMARK COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} | {'Img AUROC':<10} | {'Img AP':<8} | {'Pix AUROC':<10} | {'Throughput':<10}")
    logger.info("-" * 70)
    for name, res in baseline_results.items():
        img_auc = res["image_level"]["auroc"]
        img_ap = res["image_level"]["average_precision"]
        pix_auc = res["pixel_level"]["pixel_auroc"]
        tput = res["efficiency"]["throughput_fps"]
        logger.info(f"{name:<20} | {img_auc:<10.4f} | {img_ap:<8.4f} | {pix_auc:<10.4f} | {tput:<10.2f} FPS")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
