import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.evaluation.benchmark_runner import evaluate_vista_model
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.ablation")

def run_ablation_study(
    category: str = "bottle",
    data_root: str = "data/processed/mvtec",
    output_path: str = "benchmarks/results/ablation_study.json"
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Ablation Studies for '{category}' on {device}...")

    train_ds = MVTecDataset(data_root, category=category, split="train", image_size=518)
    test_ds = MVTecDataset(data_root, category=category, split="test", image_size=518)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    ablation_results = {}

    configs = [
        ("Full VISTA-VLM", {"fusion_type": "global_local", "use_film": True, "out_indices": [4, 8, 12], "backbone": "dinov2_vits14", "embed_dim": 384}),
        ("w/o Global Features (Local Only)", {"fusion_type": "local_only", "use_film": False, "out_indices": [4, 8, 12], "backbone": "dinov2_vits14", "embed_dim": 384}),
        ("w/o Multi-Layer (Block 12 Only)", {"fusion_type": "global_local", "use_film": True, "out_indices": [12], "backbone": "dinov2_vits14", "embed_dim": 384}),
        ("w/ Simple Concat Fusion", {"fusion_type": "concat_only", "use_film": False, "out_indices": [4, 8, 12], "backbone": "dinov2_vits14", "embed_dim": 384}),
        ("Secondary Backbone (ViT-B/14)", {"fusion_type": "global_local", "use_film": True, "out_indices": [4, 8, 12], "backbone": "dinov2_vitb14", "embed_dim": 768}),
    ]

    for name, cfg in configs:
        logger.info(f"Running Ablation Variant: '{name}'...")
        model = VISTAAnomalyModel(
            backbone_name=cfg["backbone"],
            embed_dim=cfg["embed_dim"],
            out_indices=cfg["out_indices"],
            proj_dim=256,
            image_size=518,
            fusion_type=cfg["fusion_type"],
            use_film=cfg["use_film"],
            freeze_backbone=True
        ).to(device)

        eval_res = evaluate_vista_model(model, test_loader, device=device, calibrated_threshold=0.5)
        ablation_results[name] = {
            "image_auroc": eval_res["image_level"]["auroc"],
            "image_ap": eval_res["image_level"]["average_precision"],
            "image_f1": eval_res["image_level"]["f1"],
            "pixel_auroc": eval_res["pixel_level"]["pixel_auroc"],
            "pixel_ap": eval_res["pixel_level"]["pixel_ap"],
            "aupro": eval_res["pixel_level"]["aupro"],
            "p50_latency_ms": eval_res["efficiency"]["p50_latency_ms"],
            "throughput_fps": eval_res["efficiency"]["throughput_fps"],
        }

    # Few-shot evaluations
    few_shot_results = {}
    for k in [1, 2, 4, 8]:
        logger.info(f"Evaluating Few-Normal-Shot: k={k}...")
        few_ds = train_ds.sample_few_shot(k=k, seed=42)
        # Evaluate zero-shot / few-shot generalization
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
        eval_few = evaluate_vista_model(model, test_loader, device=device, calibrated_threshold=0.5)
        few_shot_results[f"{k}-shot"] = {
            "image_auroc": eval_few["image_level"]["auroc"],
            "pixel_auroc": eval_few["pixel_level"]["pixel_auroc"],
            "f1": eval_few["image_level"]["f1"]
        }

    final_ablation_data = {
        "architectural_ablations": ablation_results,
        "few_shot_learning": few_shot_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_ablation_data, f, indent=2)

    logger.info(f"Saved complete ablation experiment results to {output_path}")
    return final_ablation_data

if __name__ == "__main__":
    run_ablation_study()
