#!/usr/bin/env python3
"""
VISTA-VLM Procedural Synthetic Fixture Generator (CI / Software Verification Only).
==================================================================================
WARNING: THIS SCRIPT GENERATES PROCEDURAL SYNTHETIC FIXTURES FOR SOFTWARE PIPELINE
AND UNIT TEST VERIFICATION ONLY. IT DOES NOT PRODUCE OR ACCESS REAL INDUSTRIAL DATA.
RESULTS GENERATED USING THESE FIXTURES MUST NEVER BE REPRESENTED AS EMPIRICAL REAL-WORLD
INDUSTRIAL INSPECTION BENCHMARKS.

Strictly writes to:
  data/synthetic_fixtures/mvtec/
  data/synthetic_fixtures/visa/
  data/manifests/synthetic/
"""

from __future__ import annotations

import os
import sys
import argparse
import shutil
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vista_vlm.data.synthetic import generate_perlin_noise_mask
from vista_vlm.data.manifest_manager import DatasetIntegrityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vista_vlm.prepare_benchmark_data")

FORBIDDEN_REAL_DATA_TOKENS = ["mvtec_ad", "real_iad", "btad", "mpdd", "visa_official"]


def validate_synthetic_output_dir(target_dir: str, real_root: str = "data/real") -> str:
    """
    Validates that target_dir is strictly safe for writing synthetic fixtures.
    Rejects any target_dir that points into, overlaps with, or is contained within real_root,
    or contains forbidden real-dataset tokens.
    """
    abs_target = os.path.realpath(os.path.abspath(target_dir))
    abs_real = os.path.realpath(os.path.abspath(real_root))

    # Reject if target is inside real_root or identical to real_root
    try:
        common = os.path.commonpath([abs_target, abs_real])
        if common == abs_real:
            raise ValueError(
                f"DATA INTEGRITY VIOLATION: Refusing to generate synthetic fixtures inside real data directory: "
                f"'{abs_target}' is located inside '{abs_real}'."
            )
    except ValueError as e:
        if "DATA INTEGRITY VIOLATION" in str(e):
            raise
        # On Windows, paths on different drives raise ValueError in commonpath, which is safe
        pass

    # Reject forbidden real dataset folder names
    parts = [p.lower() for p in abs_target.replace("\\", "/").split("/")]
    for token in FORBIDDEN_REAL_DATA_TOKENS:
        if token in parts:
            raise ValueError(
                f"DATA INTEGRITY VIOLATION: Output directory '{abs_target}' contains real dataset token '{token}'. "
                f"Synthetic fixtures must be permanently isolated under data/synthetic_fixtures/."
            )

    return abs_target


def create_synthetic_industrial_object(
    category: str,
    width: int = 518,
    height: int = 518,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generates procedural geometric/texture object fixture representing category."""
    if rng is None:
        rng = np.random.default_rng(seed)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    if category in ("bottle", "capsule"):
        img[:] = (25, 28, 32)
        cx = width // 2 + int(rng.integers(-4, 5))
        cy = height // 2 + int(rng.integers(-4, 5))
        center = (cx, cy)
        ax_w = width // 4 + int(rng.integers(-3, 4))
        ax_h = height // 3 + int(rng.integers(-4, 5))
        axes = (ax_w, ax_h)
        col = (
            int(210 + rng.integers(-5, 6)),
            int(225 + rng.integers(-5, 6)),
            int(240 + rng.integers(-5, 6))
        )
        cv2.ellipse(img, center, axes, 0, 0, 360, col, -1)
        y_min, y_max = max(0, center[1] - axes[1]), min(height, center[1] + axes[1])
        x_min, x_max = max(0, center[0] - axes[0]), min(width, center[0] + axes[0])
        patch = img[y_min:y_max, x_min:x_max]
        if patch.size > 0:
            img[y_min:y_max, x_min:x_max] = np.clip(patch * 1.05, 0, 255)

    elif category in ("cable", "transistor"):
        img[:] = (35, 35, 38)
        pts = np.array([
            [width // 4 + int(rng.integers(-5, 6)), height // 6 + int(rng.integers(-5, 6))],
            [width // 2 + int(rng.integers(-5, 6)), height // 3 + int(rng.integers(-5, 6))],
            [width // 3 + int(rng.integers(-5, 6)), 2 * height // 3 + int(rng.integers(-5, 6))],
            [3 * width // 4 + int(rng.integers(-5, 6)), 5 * height // 6 + int(rng.integers(-5, 6))]
        ], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(30, 144, 255), thickness=36)
        cv2.polylines(img, [pts], isClosed=False, color=(100, 180, 255), thickness=16)

    elif category in ("leather", "wood", "carpet", "tile"):
        base_color = (139, 90, 43) if category == "wood" else (160, 120, 90) if category == "leather" else (180, 180, 190)
        img[:] = base_color
        noise = rng.normal(0, 15, (height, width, 3)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for y in range(0, height, 16):
            cv2.line(img, (0, y), (width, y), (max(0, base_color[0]-25), max(0, base_color[1]-20), max(0, base_color[2]-15)), 1)

    elif category in ("candle", "cashew", "chewinggum", "pcb1"):
        img[:] = (240, 240, 242)
        cx = width // 2 + int(rng.integers(-4, 5))
        cy = height // 2 + int(rng.integers(-4, 5))
        center = (cx, cy)
        rad = width // 3 + int(rng.integers(-4, 5))
        cv2.circle(img, center, rad, (220, 200, 150), -1)
        cv2.circle(img, center, width // 4, (240, 220, 170), -1)

    else:
        img[:] = (40, 42, 45)
        rx = width // 4 + int(rng.integers(-4, 5))
        ry = height // 4 + int(rng.integers(-4, 5))
        rw = width // 2 + int(rng.integers(-4, 5))
        rh = height // 2 + int(rng.integers(-4, 5))
        rect = (rx, ry, rw, rh)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (180, 185, 190), -1)
        cv2.circle(img, (width // 2, height // 2), width // 8, (70, 72, 75), -1)

    # Add subtle sensor noise for realistic non-identical acquisition
    sensor_noise = rng.normal(0, 1.2, (height, width, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)
    return img


def create_synthetic_defect(
    base_img: np.ndarray,
    defect_type: str,
    width: int = 518,
    height: int = 518,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Adds a procedural defect to base image fixture and returns (defective_img, ground_truth_mask)."""
    if rng is None:
        rng = np.random.default_rng(seed)

    anom_img = base_img.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    if "scratch" in defect_type or "crack" in defect_type:
        p1 = (int(rng.integers(width // 3, 2 * width // 3)), int(rng.integers(height // 3, 2 * height // 3)))
        p2 = (int(p1[0] + rng.integers(-70, 70)), int(p1[1] + rng.integers(-70, 70)))
        thickness = int(rng.integers(3, 7))
        cv2.line(mask, p1, p2, 255, thickness=thickness)
        cv2.line(anom_img, p1, p2, (20, 20, 20), thickness=thickness)

    elif "dent" in defect_type or "hole" in defect_type:
        center = (int(rng.integers(width // 3, 2 * width // 3)), int(rng.integers(height // 3, 2 * height // 3)))
        radius = int(rng.integers(12, 35))
        cv2.circle(mask, center, radius, 255, -1)
        cv2.circle(anom_img, center, radius, (15, 15, 15), -1)

    elif "contamination" in defect_type or "stain" in defect_type:
        noise_mask = generate_perlin_noise_mask(height, width, max_scale=4, threshold=0.7)
        mask = (noise_mask * 255).astype(np.uint8)
        stain_color = np.array([40, 25, 15], dtype=np.uint8)
        idx = mask > 0
        anom_img[idx] = (anom_img[idx].astype(np.float32) * 0.3 + stain_color * 0.7).astype(np.uint8)

    else:
        cx = int(rng.integers(width // 3, 2 * width // 3))
        cy = int(rng.integers(height // 3, 2 * height // 3))
        axes = (int(rng.integers(25, 60)), int(rng.integers(20, 50)))
        angle = int(rng.integers(0, 360))
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        anom_img[mask > 0] = (255, 60, 60)

    # Ensure strictly binary mask {0, 255}
    mask = (mask > 127).astype(np.uint8) * 255
    return anom_img, mask


def prepare_mvtec_fixtures(
    output_root: str = "data/synthetic_fixtures/mvtec",
    real_root: str = "data/real",
    num_train: int = 40,
    num_test_good: int = 15,
    num_test_anom: int = 20,
    seed: int = 42
) -> str:
    """Prepares structured procedural synthetic fixtures adhering to MVTec AD directory layout."""
    safe_output = validate_synthetic_output_dir(output_root, real_root)
    os.makedirs(safe_output, exist_ok=True)
    categories = ["bottle", "cable", "leather", "capsule", "tile"]

    for cat_idx, cat in enumerate(categories):
        logger.info(f"Generating procedural synthetic regression fixtures for MVTec-like category: '{cat}'...")
        cat_dir = os.path.join(safe_output, cat)
        train_good = os.path.join(cat_dir, "train", "good")
        test_good = os.path.join(cat_dir, "test", "good")
        os.makedirs(train_good, exist_ok=True)
        os.makedirs(test_good, exist_ok=True)

        cat_seed = seed + cat_idx * 1000

        for i in range(num_train):
            img_seed = cat_seed + 100 + i
            img = create_synthetic_industrial_object(cat, seed=img_seed)
            assert img.shape == (518, 518, 3), f"Invalid image dimensions: {img.shape}"
            img_path = os.path.join(train_good, f"{i:03d}.png")
            Image.fromarray(img).save(img_path)
            # Verify file integrity
            with Image.open(img_path) as im:
                im.verify()

        for i in range(num_test_good):
            img_seed = cat_seed + 300 + i
            img = create_synthetic_industrial_object(cat, seed=img_seed)
            img_path = os.path.join(test_good, f"{i:03d}.png")
            Image.fromarray(img).save(img_path)
            with Image.open(img_path) as im:
                im.verify()

        defect_types = ["broken_large", "contamination", "scratch", "crack"]
        for dt_idx, dtype in enumerate(defect_types):
            test_dtype_dir = os.path.join(cat_dir, "test", dtype)
            gt_dtype_dir = os.path.join(cat_dir, "ground_truth", dtype)
            os.makedirs(test_dtype_dir, exist_ok=True)
            os.makedirs(gt_dtype_dir, exist_ok=True)

            count = max(3, num_test_anom // len(defect_types))
            for i in range(count):
                obj_seed = cat_seed + 500 + dt_idx * 50 + i * 2
                def_seed = cat_seed + 600 + dt_idx * 50 + i * 2 + 1
                base_img = create_synthetic_industrial_object(cat, seed=obj_seed)
                anom_img, mask = create_synthetic_defect(base_img, dtype, seed=def_seed)

                assert anom_img.shape == (518, 518, 3), f"Invalid anomaly image shape: {anom_img.shape}"
                assert mask.shape == (518, 518), f"Invalid mask shape: {mask.shape}"
                assert set(np.unique(mask)).issubset({0, 255}), f"Mask values not strictly binary: {np.unique(mask)}"

                img_path = os.path.join(test_dtype_dir, f"{i:03d}.png")
                mask_path = os.path.join(gt_dtype_dir, f"{i:03d}_mask.png")
                Image.fromarray(anom_img).save(img_path)
                Image.fromarray(mask).save(mask_path)

                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(mask_path) as im:
                    im.verify()

    return safe_output


def prepare_visa_fixtures(
    output_root: str = "data/synthetic_fixtures/visa",
    real_root: str = "data/real",
    num_normal: int = 45,
    num_anom: int = 20,
    seed: int = 42
) -> str:
    """Prepares structured procedural synthetic fixtures adhering to VisA directory layout."""
    safe_output = validate_synthetic_output_dir(output_root, real_root)
    os.makedirs(safe_output, exist_ok=True)
    categories = ["candle", "capsules", "pcb1"]

    for cat_idx, cat in enumerate(categories):
        logger.info(f"Generating procedural synthetic regression fixtures for VisA-like category: '{cat}'...")
        cat_dir = os.path.join(safe_output, cat)
        norm_dir = os.path.join(cat_dir, "Data", "Images", "Normal")
        anom_dir = os.path.join(cat_dir, "Data", "Images", "Anomaly")
        mask_dir = os.path.join(cat_dir, "Data", "Masks", "Anomaly")
        os.makedirs(norm_dir, exist_ok=True)
        os.makedirs(anom_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        cat_seed = seed + 5000 + cat_idx * 1000

        for i in range(num_normal):
            img_seed = cat_seed + 100 + i
            img = create_synthetic_industrial_object(cat, seed=img_seed)
            img_path = os.path.join(norm_dir, f"{i:04d}.png")
            Image.fromarray(img).save(img_path)
            with Image.open(img_path) as im:
                im.verify()

        for i in range(num_anom):
            obj_seed = cat_seed + 300 + i * 2
            def_seed = cat_seed + 300 + i * 2 + 1
            base_img = create_synthetic_industrial_object(cat, seed=obj_seed)
            anom_img, mask = create_synthetic_defect(base_img, "defect", seed=def_seed)

            assert anom_img.shape == (518, 518, 3)
            assert mask.shape == (518, 518)
            assert set(np.unique(mask)).issubset({0, 255})

            img_path = os.path.join(anom_dir, f"{i:04d}.png")
            mask_path = os.path.join(mask_dir, f"{i:04d}.png")
            Image.fromarray(anom_img).save(img_path)
            Image.fromarray(mask).save(mask_path)

            with Image.open(img_path) as im:
                im.verify()
            with Image.open(mask_path) as im:
                im.verify()

    return safe_output


def mirror_to_legacy_processed(
    mvtec_source: str = "data/synthetic_fixtures/mvtec",
    visa_source: str = "data/synthetic_fixtures/visa",
    real_root: str = "data/real"
):
    """Maintains backward compatibility by mirroring fixtures to data/processed with safety checks."""
    legacy_mvtec = "data/processed/mvtec"
    legacy_visa = "data/processed/visa"

    validate_synthetic_output_dir(legacy_mvtec, real_root)
    validate_synthetic_output_dir(legacy_visa, real_root)

    if os.path.exists(mvtec_source):
        os.makedirs(legacy_mvtec, exist_ok=True)
        for item in os.listdir(mvtec_source):
            s = os.path.join(mvtec_source, item)
            d = os.path.join(legacy_mvtec, item)
            if os.path.isdir(s) and not os.path.exists(d):
                shutil.copytree(s, d)

    if os.path.exists(visa_source):
        os.makedirs(legacy_visa, exist_ok=True)
        for item in os.listdir(visa_source):
            s = os.path.join(visa_source, item)
            d = os.path.join(legacy_visa, item)
            if os.path.isdir(s) and not os.path.exists(d):
                shutil.copytree(s, d)


def build_and_save_manifests(
    mvtec_dir: str = "data/synthetic_fixtures/mvtec",
    visa_dir: str = "data/synthetic_fixtures/visa",
    manifest_dir: str = "data/manifests/synthetic"
) -> Dict[str, Any]:
    """Generates immutable SHA-256 manifests for all synthetic fixtures."""
    os.makedirs(manifest_dir, exist_ok=True)

    manifest_mvtec = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_mvtec",
        root_dir=mvtec_dir,
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    DatasetIntegrityChecker.save_manifest(
        manifest_mvtec,
        os.path.join(manifest_dir, "mvtec_manifest.json")
    )
    # Also save backwards-compatible alias
    DatasetIntegrityChecker.save_manifest(
        manifest_mvtec,
        os.path.join(manifest_dir, "mvtec_like_manifest.json")
    )

    manifest_visa = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_visa",
        root_dir=visa_dir,
        provenance="procedural_synthetic_benchmark_fixture",
        layout="visa"
    )
    DatasetIntegrityChecker.save_manifest(
        manifest_visa,
        os.path.join(manifest_dir, "visa_manifest.json")
    )
    # Also save backwards-compatible alias
    DatasetIntegrityChecker.save_manifest(
        manifest_visa,
        os.path.join(manifest_dir, "visa_like_manifest.json")
    )

    logger.info(
        f"Synthetic manifests generated: mvtec ({manifest_mvtec['total_samples']} samples), "
        f"visa ({manifest_visa['total_samples']} samples)"
    )
    return {"mvtec": manifest_mvtec, "visa": manifest_visa}


def main():
    parser = argparse.ArgumentParser(
        description="VISTA-VLM Procedural Synthetic Regression Fixture Generator"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/synthetic_fixtures",
        help="Root directory for synthetic fixtures (default: data/synthetic_fixtures)"
    )
    parser.add_argument(
        "--mvtec-dir",
        type=str,
        default=None,
        help="Explicit directory for MVTec synthetic fixtures (defaults to <output-root>/mvtec)"
    )
    parser.add_argument(
        "--visa-dir",
        type=str,
        default=None,
        help="Explicit directory for VisA synthetic fixtures (defaults to <output-root>/visa)"
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests/synthetic",
        help="Directory to store generated manifests (default: data/manifests/synthetic)"
    )
    parser.add_argument(
        "--real-data-root",
        type=str,
        default="data/real",
        help="Path to real data root to protect against accidental overwrite (default: data/real)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)"
    )
    parser.add_argument(
        "--mirror-legacy",
        action="store_true",
        default=True,
        help="Mirror fixtures to data/processed for backwards compatibility (default: True)"
    )

    args = parser.parse_args()

    mvtec_dir = args.mvtec_dir or os.path.join(args.output_root, "mvtec")
    visa_dir = args.visa_dir or os.path.join(args.output_root, "visa")

    logger.info("Initializing VISTA-VLM Procedural Synthetic Fixture Preparation...")
    logger.info(f"Target MVTec fixture path: {mvtec_dir}")
    logger.info(f"Target VisA fixture path:  {visa_dir}")
    logger.info(f"Protected real data path:   {args.real_data_root}")

    prepare_mvtec_fixtures(output_root=mvtec_dir, real_root=args.real_data_root, seed=args.seed)
    prepare_visa_fixtures(output_root=visa_dir, real_root=args.real_data_root, seed=args.seed)

    if args.mirror_legacy:
        mirror_to_legacy_processed(mvtec_source=mvtec_dir, visa_source=visa_dir, real_root=args.real_data_root)

    build_and_save_manifests(mvtec_dir=mvtec_dir, visa_dir=visa_dir, manifest_dir=args.manifest_dir)
    logger.info("Synthetic fixture generation and cryptographic manifest generation complete!")


if __name__ == "__main__":
    main()
