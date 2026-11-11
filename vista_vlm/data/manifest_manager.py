"""
VISTA-VLM Dataset Integrity and Manifest Verification System (Phase 2 & Phase 3).
Computes cryptographic SHA-256 signatures, verifies split integrity,
detects train/test leakage, and generates immutable dataset manifests
with strict separation of synthetic fixtures vs real industrial datasets.
"""

from __future__ import annotations

import os
import glob
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

@dataclass
class ManifestSample:
    dataset_name: str
    category: str
    sample_id: str
    image_path: str
    image_sha256: str
    mask_path: Optional[str]
    mask_sha256: Optional[str]
    split: str  # train, val_calibration, test
    is_anomaly: bool
    defect_type: str
    width: int
    height: int
    provenance: str  # 'procedural_synthetic_benchmark_fixture', 'real_public_dataset', 'real_industrial_partner'


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 cryptographic digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DatasetIntegrityChecker:
    """
    Validates dataset integrity, checks for data leakage between train/test,
    validates mask dimensions and defective correspondence, and outputs immutable manifests.
    """

    @staticmethod
    def build_manifest(
        dataset_name: str,
        root_dir: str,
        provenance: str = "procedural_synthetic_benchmark_fixture",
        layout: str = "mvtec"  # 'mvtec' or 'visa'
    ) -> Dict[str, Any]:
        """
        Scans directory structure and constructs complete cryptographic manifest.
        """
        samples: List[ManifestSample] = []
        errors: List[str] = []
        sha_map: Dict[str, str] = {}  # sha256 -> sample_id

        if not os.path.exists(root_dir):
            return {
                "dataset_name": dataset_name,
                "root_dir": root_dir,
                "provenance": provenance,
                "status": "MISSING_ON_DISK",
                "sample_count": 0,
                "samples": [],
                "errors": [f"Directory does not exist: {root_dir}"]
            }

        categories = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ])

        for cat in categories:
            cat_dir = os.path.join(root_dir, cat)
            if layout == "mvtec":
                # Train good
                train_dir = os.path.join(cat_dir, "train", "good")
                if os.path.exists(train_dir):
                    for img_file in sorted(glob.glob(os.path.join(train_dir, "*.*"))):
                        if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            continue
                        try:
                            with Image.open(img_file) as im:
                                w, h = im.size
                            img_hash = compute_sha256(img_file)
                            sid = f"{cat}_train_{os.path.splitext(os.path.basename(img_file))[0]}"
                            samples.append(ManifestSample(
                                dataset_name=dataset_name,
                                category=cat,
                                sample_id=sid,
                                image_path=img_file,
                                image_sha256=img_hash,
                                mask_path=None,
                                mask_sha256=None,
                                split="train",
                                is_anomaly=False,
                                defect_type="good",
                                width=w,
                                height=h,
                                provenance=provenance
                            ))
                            sha_map[img_hash] = f"train:{sid}"
                        except Exception as e:
                            errors.append(f"Corrupt image {img_file}: {e}")

                # Test images
                test_dir = os.path.join(cat_dir, "test")
                gt_dir = os.path.join(cat_dir, "ground_truth")
                if os.path.exists(test_dir):
                    for dtype in sorted(os.listdir(test_dir)):
                        dt_path = os.path.join(test_dir, dtype)
                        if not os.path.isdir(dt_path):
                            continue
                        is_anom = (dtype != "good")
                        for img_file in sorted(glob.glob(os.path.join(dt_path, "*.*"))):
                            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                                continue
                            try:
                                with Image.open(img_file) as im:
                                    w, h = im.size
                                img_hash = compute_sha256(img_file)
                                sid = f"{cat}_test_{dtype}_{os.path.splitext(os.path.basename(img_file))[0]}"
                                
                                # Leakage check: test image identical to training image
                                if img_hash in sha_map and sha_map[img_hash].startswith("train:"):
                                    errors.append(f"DATA LEAKAGE DETECTED: Test image {img_file} is identical to {sha_map[img_hash]}")

                                mask_path = None
                                mask_hash = None
                                if is_anom and os.path.exists(gt_dir):
                                    fname = os.path.splitext(os.path.basename(img_file))[0]
                                    cand1 = os.path.join(gt_dir, dtype, f"{fname}_mask.png")
                                    cand2 = os.path.join(gt_dir, dtype, f"{fname}.png")
                                    if os.path.exists(cand1):
                                        mask_path = cand1
                                    elif os.path.exists(cand2):
                                        mask_path = cand2
                                    else:
                                        errors.append(f"Missing ground truth mask for anomaly: {img_file}")

                                    if mask_path:
                                        with Image.open(mask_path) as mim:
                                            mw, mh = mim.size
                                        if (mw, mh) != (w, h):
                                            errors.append(f"Dimension mismatch for {img_file} ({w}x{h}) vs mask {mask_path} ({mw}x{mh})")
                                        mask_hash = compute_sha256(mask_path)

                                samples.append(ManifestSample(
                                    dataset_name=dataset_name,
                                    category=cat,
                                    sample_id=sid,
                                    image_path=img_file,
                                    image_sha256=img_hash,
                                    mask_path=mask_path,
                                    mask_sha256=mask_hash,
                                    split="test",
                                    is_anomaly=is_anom,
                                    defect_type=dtype,
                                    width=w,
                                    height=h,
                                    provenance=provenance
                                ))
                            except Exception as e:
                                errors.append(f"Error processing test image {img_file}: {e}")

            elif layout == "visa":
                norm_dir = os.path.join(cat_dir, "Data", "Images", "Normal")
                anom_dir = os.path.join(cat_dir, "Data", "Images", "Anomaly")
                mask_dir = os.path.join(cat_dir, "Data", "Masks", "Anomaly")

                if os.path.exists(norm_dir):
                    norm_files = sorted(glob.glob(os.path.join(norm_dir, "*.*")))
                    # First 70% train, remaining test normal
                    split_idx = max(1, int(len(norm_files) * 0.7))
                    for idx, img_file in enumerate(norm_files):
                        if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            continue
                        with Image.open(img_file) as im:
                            w, h = im.size
                        img_hash = compute_sha256(img_file)
                        split = "train" if idx < split_idx else "test"
                        sid = f"{cat}_{split}_{os.path.splitext(os.path.basename(img_file))[0]}"
                        samples.append(ManifestSample(
                            dataset_name=dataset_name,
                            category=cat,
                            sample_id=sid,
                            image_path=img_file,
                            image_sha256=img_hash,
                            mask_path=None,
                            mask_sha256=None,
                            split=split,
                            is_anomaly=False,
                            defect_type="normal",
                            width=w,
                            height=h,
                            provenance=provenance
                        ))

                if os.path.exists(anom_dir):
                    anom_files = sorted(glob.glob(os.path.join(anom_dir, "*.*")))
                    for img_file in anom_files:
                        if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            continue
                        with Image.open(img_file) as im:
                            w, h = im.size
                        img_hash = compute_sha256(img_file)
                        fname = os.path.splitext(os.path.basename(img_file))[0]
                        mask_cand = os.path.join(mask_dir, f"{fname}.png")
                        mask_path = mask_cand if os.path.exists(mask_cand) else None
                        mask_hash = compute_sha256(mask_path) if mask_path else None
                        sid = f"{cat}_test_anom_{fname}"
                        samples.append(ManifestSample(
                            dataset_name=dataset_name,
                            category=cat,
                            sample_id=sid,
                            image_path=img_file,
                            image_sha256=img_hash,
                            mask_path=mask_path,
                            mask_sha256=mask_hash,
                            split="test",
                            is_anomaly=True,
                            defect_type="anomaly",
                            width=w,
                            height=h,
                            provenance=provenance
                        ))

        manifest = {
            "dataset_name": dataset_name,
            "root_dir": root_dir,
            "provenance": provenance,
            "total_samples": len(samples),
            "categories": categories,
            "integrity_errors": errors,
            "is_valid": len(errors) == 0 and len(samples) > 0,
            "samples": [asdict(s) for s in samples]
        }
        return manifest

    @staticmethod
    def save_manifest(manifest: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)


def _has_valid_images(directory: str) -> bool:
    """Helper to verify that a directory actually contains image files and is not merely an empty folder or .gitkeep."""
    if not os.path.isdir(directory):
        return False
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(valid_exts):
                return True
    return False


def resolve_dataset_root(
    dataset_key: str,
    allow_synthetic_fallback: bool = True
) -> Tuple[str, str, bool]:
    """
    Resolves dataset root path considering VISTA_DATA_ROOT, local real path, and synthetic fixtures.
    Does NOT treat folder existence or placeholder files (.gitkeep) as evidence that a real dataset is present.
    Returns:
        (resolved_path, provenance, is_real)
    """
    env_root = os.environ.get("VISTA_DATA_ROOT")
    if env_root:
        env_cand = os.path.join(env_root, dataset_key)
        if _has_valid_images(env_cand):
            return env_cand, "real_public_dataset", True

    real_path = os.path.join("data", "real", dataset_key)
    if _has_valid_images(real_path):
        return real_path, "real_public_dataset", True

    if not allow_synthetic_fallback:
        return real_path, "real_dataset_unavailable", False

    # Check synthetic fixture path (first mvtec/visa, then mvtec_like/visa_like)
    synth_path = os.path.join("data", "synthetic_fixtures", dataset_key)
    if _has_valid_images(synth_path):
        return synth_path, "procedural_synthetic_benchmark_fixture", False

    synth_like = os.path.join("data", "synthetic_fixtures", f"{dataset_key}_like")
    if _has_valid_images(synth_like):
        return synth_like, "procedural_synthetic_benchmark_fixture", False

    legacy_processed = os.path.join("data", "processed", dataset_key)
    if _has_valid_images(legacy_processed):
        return legacy_processed, "procedural_synthetic_benchmark_fixture", False

    return real_path, "unverified", False

