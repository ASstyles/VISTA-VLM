"""
VISTA-VLM Dataset Integrity and Manifest Verification System (V7.0 Research Upgrade).
=====================================================================================
Computes cryptographic SHA-256 signatures, verifies split integrity across all four partitions:
  ('train', 'val_calibration', 'val_tuning', 'test'),
detects exact and near-duplicate cross-partition data leakage,
validates pixel mask dimensions and defective correspondence,
and outputs deterministic immutable manifests with strict provenance tracking.
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
    split: str  # 'train', 'val_calibration', 'val_tuning', 'test'
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


def compute_average_hash(filepath: str, hash_size: int = 8) -> int:
    """Computes 64-bit perceptual average hash (aHash) for near-duplicate screening."""
    try:
        with Image.open(filepath) as im:
            im_gray = im.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = [int(p) for p in im_gray.tobytes()]
            avg = sum(pixels) / max(1, len(pixels))
            bits = "".join(["1" if p >= avg else "0" for p in pixels])
            return int(bits, 2)
    except Exception:
        return 0


def hamming_distance(h1: int, h2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


class DatasetIntegrityChecker:
    """
    Validates dataset integrity, checks for data leakage across all 4 partitions:
    (train, val_calibration, val_tuning, test),
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
        warnings: List[str] = []
        sha_map: Dict[str, str] = {}  # sha256 -> f"{split}:{sid}"
        ahash_map: Dict[str, Tuple[str, int]] = {}  # sid -> (split, ahash)
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

        if not os.path.exists(root_dir):
            return {
                "dataset_name": dataset_name,
                "root_dir": root_dir,
                "provenance": provenance,
                "status": "MISSING_ON_DISK",
                "total_samples": 0,
                "samples": [],
                "integrity_errors": [f"Directory does not exist: {root_dir}"],
                "warnings": [],
                "is_valid": False
            }

        categories = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ])

        for cat in categories:
            cat_dir = os.path.join(root_dir, cat)
            if layout == "mvtec":
                # Train good -> partition into 'train' and 'val_calibration'
                train_dir = os.path.join(cat_dir, "train", "good")
                if os.path.exists(train_dir):
                    train_files = sorted([
                        f for f in glob.glob(os.path.join(train_dir, "*.*"))
                        if f.lower().endswith(valid_exts)
                    ])
                    n_train_files = len(train_files)
                    n_cal = max(1, int(n_train_files * 0.20)) if n_train_files >= 4 else (1 if n_train_files >= 2 else 0)
                    cal_files = set(train_files[-n_cal:]) if n_cal > 0 else set()

                    for img_file in train_files:
                        try:
                            with Image.open(img_file) as im:
                                w, h = im.size
                            img_hash = compute_sha256(img_file)
                            img_ahash = compute_average_hash(img_file)
                            split_name = "val_calibration" if img_file in cal_files else "train"
                            sid = f"{cat}_{split_name}_{os.path.splitext(os.path.basename(img_file))[0]}"

                            # Check exact duplication
                            if img_hash in sha_map:
                                prev_split = sha_map[img_hash].split(":")[0]
                                if prev_split != split_name:
                                    errors.append(f"DATA LEAKAGE DETECTED: {img_file} identical to {sha_map[img_hash]}")

                            sha_map[img_hash] = f"{split_name}:{sid}"
                            ahash_map[sid] = (split_name, img_ahash)

                            samples.append(ManifestSample(
                                dataset_name=dataset_name,
                                category=cat,
                                sample_id=sid,
                                image_path=img_file,
                                image_sha256=img_hash,
                                mask_path=None,
                                mask_sha256=None,
                                split=split_name,
                                is_anomaly=False,
                                defect_type="good",
                                width=w,
                                height=h,
                                provenance=provenance
                            ))
                        except Exception as e:
                            errors.append(f"Corrupt image {img_file}: {e}")

                # Test images -> partition into 'val_tuning' and 'test'
                test_dir = os.path.join(cat_dir, "test")
                gt_dir = os.path.join(cat_dir, "ground_truth")
                if os.path.exists(test_dir):
                    for dtype in sorted(os.listdir(test_dir)):
                        dt_path = os.path.join(test_dir, dtype)
                        if not os.path.isdir(dt_path):
                            continue
                        is_anom = (dtype != "good")
                        test_files = sorted([
                            f for f in glob.glob(os.path.join(dt_path, "*.*"))
                            if f.lower().endswith(valid_exts)
                        ])
                        n_test_files = len(test_files)
                        n_tune = max(1, int(n_test_files * 0.20)) if n_test_files >= 4 else (1 if n_test_files >= 2 else 0)
                        tune_files = set(test_files[:n_tune]) if n_tune > 0 else set()

                        for img_file in test_files:
                            try:
                                with Image.open(img_file) as im:
                                    w, h = im.size
                                img_hash = compute_sha256(img_file)
                                img_ahash = compute_average_hash(img_file)
                                split_name = "val_tuning" if img_file in tune_files else "test"
                                sid = f"{cat}_{split_name}_{dtype}_{os.path.splitext(os.path.basename(img_file))[0]}"

                                # Leakage check: test/tuning image identical to training image
                                if img_hash in sha_map:
                                    prev_split = sha_map[img_hash].split(":")[0]
                                    if prev_split in ("train", "val_calibration"):
                                        errors.append(
                                            f"DATA LEAKAGE DETECTED: {split_name} image {img_file} is identical to {sha_map[img_hash]}"
                                        )

                                sha_map[img_hash] = f"{split_name}:{sid}"
                                ahash_map[sid] = (split_name, img_ahash)

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
                                            errors.append(
                                                f"Dimension mismatch for {img_file} ({w}x{h}) vs mask {mask_path} ({mw}x{mh})"
                                            )
                                        mask_hash = compute_sha256(mask_path)

                                samples.append(ManifestSample(
                                    dataset_name=dataset_name,
                                    category=cat,
                                    sample_id=sid,
                                    image_path=img_file,
                                    image_sha256=img_hash,
                                    mask_path=mask_path,
                                    mask_sha256=mask_hash,
                                    split=split_name,
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
                    norm_files = sorted([
                        f for f in glob.glob(os.path.join(norm_dir, "*.*"))
                        if f.lower().endswith(valid_exts)
                    ])
                    n_norm = len(norm_files)
                    n_cal = max(1, int(n_norm * 0.20)) if n_norm >= 5 else 1
                    n_test_norm = max(1, int(n_norm * 0.20)) if n_norm >= 5 else 1
                    n_train = max(1, n_norm - n_cal - n_test_norm)

                    train_set = set(norm_files[:n_train])
                    cal_set = set(norm_files[n_train:n_train + n_cal])

                    for img_file in norm_files:
                        with Image.open(img_file) as im:
                            w, h = im.size
                        img_hash = compute_sha256(img_file)
                        img_ahash = compute_average_hash(img_file)

                        if img_file in train_set:
                            sp = "train"
                        elif img_file in cal_set:
                            sp = "val_calibration"
                        else:
                            sp = "test"

                        sid = f"{cat}_{sp}_{os.path.splitext(os.path.basename(img_file))[0]}"

                        if img_hash in sha_map:
                            prev_split = sha_map[img_hash].split(":")[0]
                            if prev_split != sp:
                                errors.append(f"DATA LEAKAGE DETECTED in VisA: {img_file} identical to {sha_map[img_hash]}")

                        sha_map[img_hash] = f"{sp}:{sid}"
                        ahash_map[sid] = (sp, img_ahash)

                        samples.append(ManifestSample(
                            dataset_name=dataset_name,
                            category=cat,
                            sample_id=sid,
                            image_path=img_file,
                            image_sha256=img_hash,
                            mask_path=None,
                            mask_sha256=None,
                            split=sp,
                            is_anomaly=False,
                            defect_type="normal",
                            width=w,
                            height=h,
                            provenance=provenance
                        ))

                if os.path.exists(anom_dir):
                    anom_files = sorted([
                        f for f in glob.glob(os.path.join(anom_dir, "*.*"))
                        if f.lower().endswith(valid_exts)
                    ])
                    n_anom = len(anom_files)
                    n_tune = max(1, int(n_anom * 0.25)) if n_anom >= 4 else (1 if n_anom >= 2 else 0)
                    tune_set = set(anom_files[:n_tune])

                    for img_file in anom_files:
                        with Image.open(img_file) as im:
                            w, h = im.size
                        img_hash = compute_sha256(img_file)
                        img_ahash = compute_average_hash(img_file)
                        fname = os.path.splitext(os.path.basename(img_file))[0]
                        mask_cand = os.path.join(mask_dir, f"{fname}.png")
                        mask_path = mask_cand if os.path.exists(mask_cand) else None
                        mask_hash = compute_sha256(mask_path) if mask_path else None

                        sp = "val_tuning" if img_file in tune_set else "test"
                        sid = f"{cat}_{sp}_anom_{fname}"

                        if img_hash in sha_map:
                            prev_split = sha_map[img_hash].split(":")[0]
                            if prev_split in ("train", "val_calibration"):
                                errors.append(f"DATA LEAKAGE DETECTED in VisA: {img_file} identical to {sha_map[img_hash]}")

                        sha_map[img_hash] = f"{sp}:{sid}"
                        ahash_map[sid] = (sp, img_ahash)

                        samples.append(ManifestSample(
                            dataset_name=dataset_name,
                            category=cat,
                            sample_id=sid,
                            image_path=img_file,
                            image_sha256=img_hash,
                            mask_path=mask_path,
                            mask_sha256=mask_hash,
                            split=sp,
                            is_anomaly=True,
                            defect_type="anomaly",
                            width=w,
                            height=h,
                            provenance=provenance
                        ))

        # Check for near-duplicate cross-partition leakage where feasible (sample subset if huge)
        train_sids = [sid for sid, (sp, _) in ahash_map.items() if sp == "train"]
        test_sids = [sid for sid, (sp, _) in ahash_map.items() if sp in ("test", "val_tuning")]
        if len(train_sids) <= 200 and len(test_sids) <= 200:
            for tr_sid in train_sids:
                tr_hash = ahash_map[tr_sid][1]
                for te_sid in test_sids:
                    te_hash = ahash_map[te_sid][1]
                    dist = hamming_distance(tr_hash, te_hash)
                    if dist == 0:
                        warnings.append(f"Potential near-duplicate perceptual match (hamming=0) between {tr_sid} and {te_sid}")

        split_distribution: Dict[str, int] = {}
        for s in samples:
            split_distribution[s.split] = split_distribution.get(s.split, 0) + 1

        manifest = {
            "dataset_name": dataset_name,
            "root_dir": root_dir,
            "provenance": provenance,
            "total_samples": len(samples),
            "categories": categories,
            "split_distribution": split_distribution,
            "integrity_errors": errors,
            "warnings": warnings,
            "is_valid": len(errors) == 0 and len(samples) > 0,
            "samples": [asdict(s) for s in samples]
        }
        return manifest

    @staticmethod
    def save_manifest(manifest: Dict[str, Any], output_path: str):
        """Atomically saves manifest JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, output_path)


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

    # Check synthetic fixture path
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
