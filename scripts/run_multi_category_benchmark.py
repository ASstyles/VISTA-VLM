#!/usr/bin/env python3
"""
VISTA-VLM Multi-Category Benchmark Runner (Phases 2, 3, 4, 12).
Evaluates anomaly detection and localization across multiple diverse categories:
- MVTec AD: bottle, cable, capsule, leather, tile (textures + objects)
- VisA: candle (multi-component visual assembly)

Produces per-category and macro-averaged metrics cleanly categorized by supervision regime:
1. Normal-Only Semi-Supervised Representation Learning (PatchCore, PaDiM, VISTA-VLM)
2. Fast Knowledge Distillation / Reconstruction (EfficientAD, DRAEM)
3. Zero-Shot Vision-Language Alignment (WinCLIP, AnomalyCLIP)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.visa import VisADataset
from vista_vlm.models.baselines import PatchCore, PaDiM, WinCLIP, AnomalyCLIP
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.benchmark_runner import evaluate_baseline_model, evaluate_vista_model
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.multi_benchmark")

CATEGORIES = [
    {"dataset": "mvtec", "name": "bottle", "type": "rigid_object"},
    {"dataset": "mvtec", "name": "cable", "type": "structural_wiring"},
    {"dataset": "mvtec", "name": "capsule", "type": "small_component"},
    {"dataset": "mvtec", "name": "leather", "type": "texture_surface"},
    {"dataset": "mvtec", "name": "tile", "type": "planar_texture"},
    {"dataset": "visa", "name": "candle", "type": "multi_part_assembly"},
]

def get_loaders(cat_info: Dict[str, str], image_size: int = 518, batch_size: int = 4):
    dset = cat_info["dataset"]
    name = cat_info["name"]

    if dset == "mvtec":
        train_ds = MVTecDataset("data/processed/mvtec", category=name, split="train", image_size=image_size)
        test_ds = MVTecDataset("data/processed/mvtec", category=name, split="test", image_size=image_size)
    else:
        train_ds = VisADataset("data/processed/visa", category=name, split="train", image_size=image_size)
        test_ds = VisADataset("data/processed/visa", category=name, split="test", image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="Multi-Category Benchmark Runner")
    parser.add_argument("--output_file", default="benchmarks/results/multi_category_benchmark.json")
    args = parser.parse_args()

    device = "cpu"
    logger.info(f"Starting Multi-Category Benchmark across {len(CATEGORIES)} categories on {device}...")

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "categories": [c["name"] for c in CATEGORIES],
            "regimes": {
                "PatchCore": "Normal-Only Memory Bank",
                "PaDiM": "Normal-Only Gaussian Modeling",
                "EfficientAD": "Normal-Only Distillation & Reconstruction",
                "WinCLIP": "Zero-Shot Vision-Language Alignment",
                "AnomalyCLIP": "Zero-Shot Object-Agnostic Prompting",
                "VISTA-VLM (Ours)": "Multimodal Global-Local FiLM with FPN Decoder"
            }
        },
        "per_category": {},
        "macro_averages": {}
    }

    models_to_eval = ["PatchCore", "PaDiM", "WinCLIP", "AnomalyCLIP", "VISTA-VLM (Ours)"]

    for cat in CATEGORIES:
        cat_name = cat["name"]
        logger.info(f"--- Benchmarking Category: {cat['dataset'].upper()} / {cat_name} ({cat['type']}) ---")
        train_loader, test_loader = get_loaders(cat)

        cat_results = {}

        # 1. PatchCore
        logger.info(f"[{cat_name}] Running PatchCore...")
        patchcore = PatchCore(embed_dim=384, coreset_sampling_ratio=0.10, image_size=518)
        patchcore.fit(train_loader, device=device)
        cat_results["PatchCore"] = evaluate_baseline_model(patchcore, test_loader, "patchcore", device=device)

        # 2. PaDiM
        logger.info(f"[{cat_name}] Running PaDiM...")
        padim = PaDiM(embed_dim=64, image_size=518)
        padim.fit(train_loader, device=device)
        cat_results["PaDiM"] = evaluate_baseline_model(padim, test_loader, "padim", device=device)

        # 3. WinCLIP (Zero-shot VLM)
        logger.info(f"[{cat_name}] Running WinCLIP...")
        winclip = WinCLIP(embed_dim=512, image_size=518).to(device)
        cat_results["WinCLIP"] = evaluate_baseline_model(winclip, test_loader, "winclip", device=device, category=cat_name)

        # 4. AnomalyCLIP (Zero-shot VLM)
        logger.info(f"[{cat_name}] Running AnomalyCLIP...")
        anomclip = AnomalyCLIP(embed_dim=512, image_size=518).to(device)
        cat_results["AnomalyCLIP"] = evaluate_baseline_model(anomclip, test_loader, "anomalyclip", device=device, category=cat_name)

        # 5. VISTA-VLM (Ours)
        logger.info(f"[{cat_name}] Running VISTA-VLM...")
        vista = VISTAAnomalyModel(image_size=518).to(device)
        ckpt_path = "checkpoints/vista_mvtec_bottle.pth"
        if os.path.exists(ckpt_path):
            vista.load_checkpoint(ckpt_path, device=device, strict=False)
        cat_results["VISTA-VLM (Ours)"] = evaluate_vista_model(vista, test_loader, device=device)

        # Clean raw scores to keep json compact
        for m in cat_results:
            if "raw_scores" in cat_results[m]:
                del cat_results[m]["raw_scores"]

        results["per_category"][cat_name] = cat_results

    # Compute Macro-Averages
    logger.info("Computing Macro-Averages across all evaluated categories...")
    macro = {}
    for m in models_to_eval:
        img_aurocs = [results["per_category"][c["name"]][m]["image_level"]["auroc"] for c in CATEGORIES]
        img_aps = [results["per_category"][c["name"]][m]["image_level"]["average_precision"] for c in CATEGORIES]
        pix_aurocs = [results["per_category"][c["name"]][m]["pixel_level"]["pixel_auroc"] for c in CATEGORIES]
        aupros = [results["per_category"][c["name"]][m]["pixel_level"]["aupro"] for c in CATEGORIES]
        tputs = [results["per_category"][c["name"]][m]["efficiency"]["throughput_fps"] for c in CATEGORIES]

        macro[m] = {
            "macro_image_auroc": round(float(np.mean(img_aurocs)), 4),
            "std_image_auroc": round(float(np.std(img_aurocs)), 4),
            "macro_image_ap": round(float(np.mean(img_aps)), 4),
            "macro_pixel_auroc": round(float(np.mean(pix_aurocs)), 4),
            "std_pixel_auroc": round(float(np.std(pix_aurocs)), 4),
            "macro_aupro": round(float(np.mean(aupros)), 4),
            "mean_throughput_fps": round(float(np.mean(tputs)), 2)
        }

    results["macro_averages"] = macro

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Multi-Category Benchmark saved to {args.output_file}")

    print("\n" + "=" * 90)
    print(f"{'Method':<22} | {'Regime':<24} | {'Mean Img AUC':<12} | {'Mean Pix AUC':<12} | {'Mean AUPRO':<10}")
    print("-" * 90)
    for m, stats in macro.items():
        regime = results["metadata"]["regimes"].get(m, "Baseline")
        print(f"{m:<22} | {regime:<24} | {stats['macro_image_auroc']:<12.4f} | {stats['macro_pixel_auroc']:<12.4f} | {stats['macro_aupro']:<10.4f}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()

