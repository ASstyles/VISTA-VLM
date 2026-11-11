#!/usr/bin/env python3
"""
VISTA-VLM Procedural Synthetic Fixture Generator (CI / Software Verification Only).
==================================================================================
WARNING: THIS SCRIPT GENERATES PROCEDURAL SYNTHETIC FIXTURES FOR SOFTWARE PIPELINE
AND UNIT TEST VERIFICATION ONLY. IT DOES NOT PRODUCE OR ACCESS REAL INDUSTRIAL DATA.
RESULTS GENERATED USING THESE FIXTURES MUST NEVER BE REPRESENTED AS EMPIRICAL REAL-WORLD
INDUSTRIAL INSPECTION BENCHMARKS.

Strictly writes to:
  data/synthetic_fixtures/mvtec_like/
  data/synthetic_fixtures/visa_like/
  data/manifests/synthetic/synthetic_manifest.json
"""

import os
import sys
import shutil
import numpy as np
import cv2
from PIL import Image
import json
import logging

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista_vlm.data.synthetic import generate_perlin_noise_mask, generate_geometric_mask
from vista_vlm.data.manifest_manager import DatasetIntegrityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vista_vlm.prepare_benchmark_data")


def create_synthetic_industrial_object(category: str, width: int = 518, height: int = 518, seed: int = 0) -> np.ndarray:
    """Generates procedural geometric/texture object fixture representing category."""
    np.random.seed(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if category in ("bottle", "capsule"):
        img[:] = (25, 28, 32)
        cx = width // 2 + int(np.random.randint(-4, 5))
        cy = height // 2 + int(np.random.randint(-4, 5))
        center = (cx, cy)
        ax_w = width // 4 + int(np.random.randint(-3, 4))
        ax_h = height // 3 + int(np.random.randint(-4, 5))
        axes = (ax_w, ax_h)
        col = (int(210 + np.random.randint(-5, 6)), int(225 + np.random.randint(-5, 6)), int(240 + np.random.randint(-5, 6)))
        cv2.ellipse(img, center, axes, 0, 0, 360, col, -1)
        # Add highlight gradient with subtle seed variation
        y_min, y_max = max(0, center[1] - axes[1]), min(height, center[1] + axes[1])
        x_min, x_max = max(0, center[0] - axes[0]), min(width, center[0] + axes[0])
        patch = img[y_min:y_max, x_min:x_max]
        if patch.size > 0:
            img[y_min:y_max, x_min:x_max] = np.clip(patch * 1.05, 0, 255)

    elif category in ("cable", "transistor"):
        img[:] = (35, 35, 38)
        pts = np.array([
            [width // 4 + int(np.random.randint(-5, 6)), height // 6 + int(np.random.randint(-5, 6))],
            [width // 2 + int(np.random.randint(-5, 6)), height // 3 + int(np.random.randint(-5, 6))],
            [width // 3 + int(np.random.randint(-5, 6)), 2 * height // 3 + int(np.random.randint(-5, 6))],
            [3 * width // 4 + int(np.random.randint(-5, 6)), 5 * height // 6 + int(np.random.randint(-5, 6))]
        ], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(30, 144, 255), thickness=36)
        cv2.polylines(img, [pts], isClosed=False, color=(100, 180, 255), thickness=16)

    elif category in ("leather", "wood", "carpet", "tile"):
        base_color = (139, 90, 43) if category == "wood" else (160, 120, 90) if category == "leather" else (180, 180, 190)
        img[:] = base_color
        noise = np.random.normal(0, 15, (height, width, 3)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for y in range(0, height, 16):
            cv2.line(img, (0, y), (width, y), (max(0, base_color[0]-25), max(0, base_color[1]-20), max(0, base_color[2]-15)), 1)

    elif category in ("candle", "cashew", "chewinggum", "pcb1"):
        img[:] = (240, 240, 242)
        cx = width // 2 + int(np.random.randint(-4, 5))
        cy = height // 2 + int(np.random.randint(-4, 5))
        center = (cx, cy)
        rad = width // 3 + int(np.random.randint(-4, 5))
        cv2.circle(img, center, rad, (220, 200, 150), -1)
        cv2.circle(img, center, width // 4, (240, 220, 170), -1)

    else:
        img[:] = (40, 42, 45)
        rx = width // 4 + int(np.random.randint(-4, 5))
        ry = height // 4 + int(np.random.randint(-4, 5))
        rw = width // 2 + int(np.random.randint(-4, 5))
        rh = height // 2 + int(np.random.randint(-4, 5))
        rect = (rx, ry, rw, rh)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (180, 185, 190), -1)
        cv2.circle(img, (width // 2, height // 2), width // 8, (70, 72, 75), -1)

    # Add subtle sensor noise for realistic non-identical acquisition
    sensor_noise = np.random.normal(0, 1.2, (height, width, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)
    return img


def create_synthetic_defect(
    base_img: np.ndarray, 
    defect_type: str, 
    width: int = 518, 
    height: int = 518,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Adds a procedural defect to base image fixture and returns (defective_img, ground_truth_mask)."""
    np.random.seed(seed)
    anom_img = base_img.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    if "scratch" in defect_type or "crack" in defect_type:
        p1 = (np.random.randint(width // 3, 2 * width // 3), np.random.randint(height // 3, 2 * height // 3))
        p2 = (p1[0] + np.random.randint(-70, 70), p1[1] + np.random.randint(-70, 70))
        cv2.line(mask, p1, p2, 255, thickness=np.random.randint(3, 7))
        cv2.line(anom_img, p1, p2, (20, 20, 20), thickness=np.random.randint(3, 7))

    elif "dent" in defect_type or "hole" in defect_type:
        center = (np.random.randint(width // 3, 2 * width // 3), np.random.randint(height // 3, 2 * height // 3))
        radius = np.random.randint(12, 35)
        cv2.circle(mask, center, radius, 255, -1)
        cv2.circle(anom_img, center, radius, (15, 15, 15), -1)

    elif "contamination" in defect_type or "stain" in defect_type:
        noise_mask = generate_perlin_noise_mask(height, width, max_scale=4, threshold=0.7)
        mask = (noise_mask * 255).astype(np.uint8)
        stain_color = np.array([40, 25, 15], dtype=np.uint8)
        idx = mask > 0
        anom_img[idx] = (anom_img[idx].astype(np.float32) * 0.3 + stain_color * 0.7).astype(np.uint8)

    else:
        cx = np.random.randint(width // 3, 2 * width // 3)
        cy = np.random.randint(height // 3, 2 * height // 3)
        axes = (np.random.randint(25, 60), np.random.randint(20, 50))
        angle = np.random.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        anom_img[mask > 0] = (255, 60, 60)

    mask = (mask > 127).astype(np.uint8) * 255
    return anom_img, mask


def prepare_mvtec_fixtures(
    output_root: str = "data/synthetic_fixtures/mvtec_like",
    num_train: int = 25,
    num_test_good: int = 10,
    num_test_anom: int = 15
):
    """Prepares structured procedural synthetic fixtures adhering to MVTec AD directory layout."""
    os.makedirs(output_root, exist_ok=True)
    categories = ["bottle", "cable", "leather", "capsule", "tile"]

    for cat in categories:
        logger.info(f"Generating synthetic fixture for MVTec-like category: '{cat}'...")
        cat_dir = os.path.join(output_root, cat)
        train_good = os.path.join(cat_dir, "train", "good")
        test_good = os.path.join(cat_dir, "test", "good")
        os.makedirs(train_good, exist_ok=True)
        os.makedirs(test_good, exist_ok=True)

        for i in range(num_train):
            img = create_synthetic_industrial_object(cat, seed=1000 + i)
            Image.fromarray(img).save(os.path.join(train_good, f"{i:03d}.png"))

        for i in range(num_test_good):
            img = create_synthetic_industrial_object(cat, seed=2000 + i)
            Image.fromarray(img).save(os.path.join(test_good, f"{i:03d}.png"))

        defect_types = ["broken_large", "contamination", "scratch", "crack"]
        for dtype in defect_types:
            test_dtype_dir = os.path.join(cat_dir, "test", dtype)
            gt_dtype_dir = os.path.join(cat_dir, "ground_truth", dtype)
            os.makedirs(test_dtype_dir, exist_ok=True)
            os.makedirs(gt_dtype_dir, exist_ok=True)

            for i in range(max(3, num_test_anom // len(defect_types))):
                base_img = create_synthetic_industrial_object(cat, seed=3000 + i * 10)
                anom_img, mask = create_synthetic_defect(base_img, dtype, seed=4000 + i * 10)
                Image.fromarray(anom_img).save(os.path.join(test_dtype_dir, f"{i:03d}.png"))
                Image.fromarray(mask).save(os.path.join(gt_dtype_dir, f"{i:03d}_mask.png"))


def prepare_visa_fixtures(
    output_root: str = "data/synthetic_fixtures/visa_like",
    num_normal: int = 30,
    num_anom: int = 15
):
    """Prepares structured procedural synthetic fixtures adhering to VisA directory layout."""
    os.makedirs(output_root, exist_ok=True)
    categories = ["candle", "capsules", "pcb1"]

    for cat in categories:
        logger.info(f"Generating synthetic fixture for VisA-like category: '{cat}'...")
        cat_dir = os.path.join(output_root, cat)
        norm_dir = os.path.join(cat_dir, "Data", "Images", "Normal")
        anom_dir = os.path.join(cat_dir, "Data", "Images", "Anomaly")
        mask_dir = os.path.join(cat_dir, "Data", "Masks", "Anomaly")
        os.makedirs(norm_dir, exist_ok=True)
        os.makedirs(anom_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(num_normal):
            img = create_synthetic_industrial_object(cat, seed=5000 + i)
            Image.fromarray(img).save(os.path.join(norm_dir, f"{i:04d}.png"))

        for i in range(num_anom):
            base_img = create_synthetic_industrial_object(cat, seed=6000 + i)
            anom_img, mask = create_synthetic_defect(base_img, "defect", seed=7000 + i)
            Image.fromarray(anom_img).save(os.path.join(anom_dir, f"{i:04d}.png"))
            Image.fromarray(mask).save(os.path.join(mask_dir, f"{i:04d}.png"))


def mirror_to_legacy_processed():
    """Maintains backward compatibility by mirroring fixtures to data/processed with provenance tags."""
    legacy_mvtec = "data/processed/mvtec"
    legacy_visa = "data/processed/visa"
    synth_mvtec = "data/synthetic_fixtures/mvtec_like"
    synth_visa = "data/synthetic_fixtures/visa_like"

    if os.path.exists(synth_mvtec):
        os.makedirs(legacy_mvtec, exist_ok=True)
        for item in os.listdir(synth_mvtec):
            s = os.path.join(synth_mvtec, item)
            d = os.path.join(legacy_mvtec, item)
            if os.path.isdir(s) and not os.path.exists(d):
                shutil.copytree(s, d)

    if os.path.exists(synth_visa):
        os.makedirs(legacy_visa, exist_ok=True)
        for item in os.listdir(synth_visa):
            s = os.path.join(synth_visa, item)
            d = os.path.join(legacy_visa, item)
            if os.path.isdir(s) and not os.path.exists(d):
                shutil.copytree(s, d)


def build_and_save_manifests():
    """Generates immutable SHA-256 manifests for all synthetic fixtures."""
    os.makedirs("data/manifests/synthetic", exist_ok=True)
    manifest_mvtec = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_mvtec_like",
        root_dir="data/synthetic_fixtures/mvtec_like",
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    DatasetIntegrityChecker.save_manifest(manifest_mvtec, "data/manifests/synthetic/mvtec_like_manifest.json")

    manifest_visa = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_visa_like",
        root_dir="data/synthetic_fixtures/visa_like",
        provenance="procedural_synthetic_benchmark_fixture",
        layout="visa"
    )
    DatasetIntegrityChecker.save_manifest(manifest_visa, "data/manifests/synthetic/visa_like_manifest.json")

    logger.info(f"Synthetic manifests generated: mvtec_like ({manifest_mvtec['total_samples']} samples), visa_like ({manifest_visa['total_samples']} samples)")


def main():
    logger.info("Initializing VISTA-VLM Procedural Synthetic Fixture Preparation...")
    prepare_mvtec_fixtures()
    prepare_visa_fixtures()
    mirror_to_legacy_processed()
    build_and_save_manifests()
    logger.info("Synthetic fixture generation and cryptographic manifest generation complete!")


if __name__ == "__main__":
    main()
