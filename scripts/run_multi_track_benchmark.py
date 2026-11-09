#!/usr/bin/env python3
"""
VISTA-VLM Multi-Track Benchmark Suite (v3.0.0).
Rigorously separates benchmark results into 4 independent evaluation tracks:

Track 1: In-Domain Normal-Only Industrial Inspection
  - Trained/fitted on target category normal images.
  - PatchCore, PaDiM, VISTA-VLM (In-Domain bottle).

Track 2: Source-Only Cross-Category Transfer
  - VISTA-VLM trained on source category (bottle), evaluated zero-shot
    across unseen target categories (cable, capsule, leather, tile, candle).
  - Measures cross-domain zero-shot defect transfer.

Track 3: Zero-Shot Vision-Language Alignment
  - No normal target training images. Textual prompt ensembles.
  - WinCLIP, AnomalyCLIP.

Track 4: Few-Normal-Shot Sample Efficiency
  - Sample-constrained regime with k in {1, 2, 4, 8} normal reference images.
  - PatchCore, PaDiM.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.visa import VisADataset
from vista_vlm.models.baselines import PatchCore, PaDiM, WinCLIP, AnomalyCLIP
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.benchmark_runner import evaluate_baseline_model, evaluate_vista_model
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.multi_track_benchmark")

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
    return train_ds, train_loader, test_loader

def clean_result_dict(res: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(res)
    if "raw_scores" in cleaned:
        del cleaned["raw_scores"]
    return cleaned

def compute_macro_averages(per_cat: Dict[str, Dict[str, Any]], model_names: List[str], cat_names: List[str]) -> Dict[str, Any]:
    macro = {}
    for m in model_names:
        img_aurocs = []
        img_aps = []
        pix_aurocs = []
        aupros = []
        tputs = []
        for c in cat_names:
            if c in per_cat and m in per_cat[c]:
                r = per_cat[c][m]
                img_aurocs.append(r["image_level"]["auroc"])
                img_aps.append(r["image_level"]["average_precision"])
                pix_aurocs.append(r["pixel_level"]["pixel_auroc"])
                aupros.append(r["pixel_level"]["aupro"])
                tputs.append(r["efficiency"]["throughput_fps"])

        if img_aurocs:
            macro[m] = {
                "macro_image_auroc": round(float(np.mean(img_aurocs)), 4),
                "std_image_auroc": round(float(np.std(img_aurocs)), 4),
                "macro_image_ap": round(float(np.mean(img_aps)), 4),
                "macro_pixel_auroc": round(float(np.mean(pix_aurocs)), 4),
                "std_pixel_auroc": round(float(np.std(pix_aurocs)), 4),
                "macro_aupro": round(float(np.mean(aupros)), 4),
                "mean_throughput_fps": round(float(np.mean(tputs)), 2)
            }
    return macro

def main():
    parser = argparse.ArgumentParser(description="Multi-Track Benchmark Suite")
    parser.add_argument("--output_file", default="benchmarks/results/multi_track_benchmark.json")
    args = parser.parse_args()

    device = "cpu"
    logger.info(f"Launching VISTA-VLM v3.0 Multi-Track Benchmark Suite on {device}...")

    # Load calibrated VISTA model
    vista_bottle = VISTAAnomalyModel(image_size=518).to(device)
    ckpt_path = "checkpoints/vista_mvtec_bottle.pth"
    if os.path.exists(ckpt_path):
        vista_bottle.load_checkpoint(ckpt_path, device=device, strict=False)
        logger.info(f"Loaded VISTA v3.0 checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Running with initialized weights.")

    cal_path = "checkpoints/calibration_mvtec.json"
    bottle_threshold = 0.5
    if os.path.exists(cal_path):
        with open(cal_path, "r") as f:
            cal_data = json.load(f)
            bottle_threshold = cal_data.get("category_thresholds", {}).get("bottle", 0.5)
        logger.info(f"Loaded calibrated threshold for bottle: {bottle_threshold:.4f}")

    benchmark_data = {
        "metadata": {
            "version": "3.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "categories": [c["name"] for c in CATEGORIES],
            "track_definitions": {
                "track_1_in_domain": "Trained/fitted on target category normal images. PatchCore, PaDiM, VISTA-VLM (bottle).",
                "track_2_cross_category": "Source-only transfer: VISTA-VLM trained on bottle, evaluated zero-shot on unseen target categories.",
                "track_3_zero_shot_vlm": "Zero target training images. Language-guided prompt ensembles (WinCLIP, AnomalyCLIP).",
                "track_4_few_shot": "Sample-constrained few-normal-shot adaptation (k in {1, 2, 4, 8})."
            }
        },
        "track_1_in_domain": {"per_category": {}, "macro_averages": {}},
        "track_2_cross_category": {"source": "bottle", "per_category": {}, "macro_averages": {}},
        "track_3_zero_shot_vlm": {"per_category": {}, "macro_averages": {}},
        "track_4_few_shot": {"category": "bottle", "results": {}}
    }

    # Cache dataset loaders
    loaders = {}
    for cat in CATEGORIES:
        cname = cat["name"]
        train_ds, train_l, test_l = get_loaders(cat)
        loaders[cname] = {"train_ds": train_ds, "train_loader": train_l, "test_loader": test_l, "info": cat}

    # =========================================================================
    # TRACK 1: In-Domain Normal-Only Industrial Inspection
    # =========================================================================
    logger.info("=== Running Track 1: In-Domain Normal-Only Industrial Inspection ===")
    t1_models = ["PatchCore", "PaDiM"]
    for cat in CATEGORIES:
        cname = cat["name"]
        t_load = loaders[cname]["train_loader"]
        te_load = loaders[cname]["test_loader"]
        cat_res = {}

        # 1. PatchCore
        logger.info(f"[Track 1][{cname}] Fitting & Evaluating PatchCore...")
        patchcore = PatchCore(embed_dim=384, coreset_sampling_ratio=0.10, image_size=518)
        patchcore.fit(t_load, device=device)
        cat_res["PatchCore"] = clean_result_dict(evaluate_baseline_model(patchcore, te_load, "patchcore", device=device))

        # 2. PaDiM
        logger.info(f"[Track 1][{cname}] Fitting & Evaluating PaDiM...")
        padim = PaDiM(embed_dim=64, image_size=518)
        padim.fit(t_load, device=device)
        cat_res["PaDiM"] = clean_result_dict(evaluate_baseline_model(padim, te_load, "padim", device=device))

        # 3. VISTA-VLM (In-Domain on bottle)
        if cname == "bottle":
            logger.info(f"[Track 1][bottle] Evaluating VISTA-VLM (In-Domain Bottle)...")
            cat_res["VISTA-VLM (In-Domain)"] = clean_result_dict(
                evaluate_vista_model(vista_bottle, te_load, device=device, calibrated_threshold=bottle_threshold)
            )

        benchmark_data["track_1_in_domain"]["per_category"][cname] = cat_res

    all_cat_names = [c["name"] for c in CATEGORIES]
    benchmark_data["track_1_in_domain"]["macro_averages"] = compute_macro_averages(
        benchmark_data["track_1_in_domain"]["per_category"],
        ["PatchCore", "PaDiM", "VISTA-VLM (In-Domain)"],
        all_cat_names
    )

    # =========================================================================
    # TRACK 2: Source-Only Cross-Category Transfer (VISTA-VLM Bottle -> Unseen)
    # =========================================================================
    logger.info("=== Running Track 2: Source-Only Cross-Category Transfer ===")
    unseen_cats = [c for c in CATEGORIES if c["name"] != "bottle"]
    for cat in unseen_cats:
        cname = cat["name"]
        te_load = loaders[cname]["test_loader"]
        logger.info(f"[Track 2] Evaluating VISTA-VLM (Bottle Source) -> {cname} (Target Zero-Shot)...")
        res = clean_result_dict(evaluate_vista_model(vista_bottle, te_load, device=device))
        benchmark_data["track_2_cross_category"]["per_category"][cname] = {
            "VISTA-VLM (Source: Bottle)": res
        }

    unseen_cat_names = [c["name"] for c in unseen_cats]
    benchmark_data["track_2_cross_category"]["macro_averages"] = compute_macro_averages(
        benchmark_data["track_2_cross_category"]["per_category"],
        ["VISTA-VLM (Source: Bottle)"],
        unseen_cat_names
    )

    # =========================================================================
    # TRACK 3: Zero-Shot Vision-Language Alignment (WinCLIP, AnomalyCLIP)
    # =========================================================================
    logger.info("=== Running Track 3: Zero-Shot Vision-Language Alignment ===")
    winclip = WinCLIP(embed_dim=512, image_size=518).to(device)
    anomclip = AnomalyCLIP(embed_dim=512, image_size=518).to(device)

    for cat in CATEGORIES:
        cname = cat["name"]
        te_load = loaders[cname]["test_loader"]
        cat_res = {}

        logger.info(f"[Track 3][{cname}] Evaluating WinCLIP...")
        cat_res["WinCLIP"] = clean_result_dict(evaluate_baseline_model(winclip, te_load, "winclip", device=device, category=cname))

        logger.info(f"[Track 3][{cname}] Evaluating AnomalyCLIP...")
        cat_res["AnomalyCLIP"] = clean_result_dict(evaluate_baseline_model(anomclip, te_load, "anomalyclip", device=device, category=cname))

        benchmark_data["track_3_zero_shot_vlm"]["per_category"][cname] = cat_res

    benchmark_data["track_3_zero_shot_vlm"]["macro_averages"] = compute_macro_averages(
        benchmark_data["track_3_zero_shot_vlm"]["per_category"],
        ["WinCLIP", "AnomalyCLIP"],
        all_cat_names
    )

    # =========================================================================
    # TRACK 4: Few-Normal-Shot Sample Efficiency (k in {1, 2, 4, 8})
    # =========================================================================
    logger.info("=== Running Track 4: Few-Normal-Shot Sample Efficiency on Bottle ===")
    k_shots = [1, 2, 4, 8]
    bottle_train_ds = loaders["bottle"]["train_ds"]
    bottle_test_l = loaders["bottle"]["test_loader"]

    np.random.seed(42)
    indices = np.random.permutation(len(bottle_train_ds))

    for k in k_shots:
        logger.info(f"[Track 4] Evaluating Few-Shot k={k}...")
        sub_indices = indices[:k].tolist()
        few_ds = Subset(bottle_train_ds, sub_indices)
        few_loader = DataLoader(few_ds, batch_size=min(k, 4), shuffle=False)

        # PatchCore k-shot
        pc_k = PatchCore(embed_dim=384, coreset_sampling_ratio=1.0, image_size=518)
        pc_k.fit(few_loader, device=device)
        pc_res = clean_result_dict(evaluate_baseline_model(pc_k, bottle_test_l, "patchcore", device=device))

        # PaDiM k-shot
        padim_k = PaDiM(embed_dim=64, image_size=518)
        padim_k.fit(few_loader, device=device)
        padim_res = clean_result_dict(evaluate_baseline_model(padim_k, bottle_test_l, "padim", device=device))

        benchmark_data["track_4_few_shot"]["results"][f"{k}-shot"] = {
            "PatchCore": {
                "image_auroc": pc_res["image_level"]["auroc"],
                "pixel_auroc": pc_res["pixel_level"]["pixel_auroc"],
                "aupro": pc_res["pixel_level"]["aupro"]
            },
            "PaDiM": {
                "image_auroc": padim_res["image_level"]["auroc"],
                "pixel_auroc": padim_res["pixel_level"]["pixel_auroc"],
                "aupro": padim_res["pixel_level"]["aupro"]
            }
        }

    # Save complete benchmark json
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"Multi-Track Benchmark results successfully saved to {args.output_file}")

    # Also update multi_category_benchmark.json for backward compatibility
    legacy_output = "benchmarks/results/multi_category_benchmark.json"
    with open(legacy_output, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"Updated {legacy_output} for dashboard compatibility.")

if __name__ == "__main__":
    main()
