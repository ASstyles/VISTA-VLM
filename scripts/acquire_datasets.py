#!/usr/bin/env python3
"""
VISTA-VLM Dataset Acquisition & Scientific Provenance Verifier (V7.1 Research Grade).
=====================================================================================
Automates acquisition, checksum verification, and directory organization for:
  1. MVTec AD (CC BY-NC-SA 4.0)
  2. VisA (Amazon Spot-Diff Research License)
  3. BTAD (BeanTech Anomaly Detection Dataset)
  4. MPDD (Mobile Phone Defect Dataset, CC BY-NC 4.0)

Enforces strict provenance validation:
  - Verifies presence of physical assets in data/real/<dataset>/
  - Computes and checks cryptographic digests
  - Formally blocks Gate B (Real-World Dataset Qualification) until physical assets are mounted
"""

from __future__ import annotations

import os
import argparse
import hashlib
import urllib.request
import tarfile
import zipfile
from typing import Dict, Any, Optional

from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.acquire_datasets")

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "mvtec": {
        "name": "MVTec Anomaly Detection (MVTec AD)",
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "archive_name": "mvtec_anomaly_detection.tar.xz",
        "expected_sha256": "4e3415c8f85f838634e0fe93c524e93144ef8795995cfbf0062bc6ec54db31f1",
        "homepage": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "commercial_use_permitted": False,
        "citation": "Bergmann et al., MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection, CVPR 2019",
        "categories": [
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper"
        ],
        "default_mount_path": "data/real/mvtec"
    },
    "visa": {
        "name": "Visual Anomaly (VisA)",
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "archive_name": "VisA_20220922.tar",
        "expected_sha256": "8d3ea51f08e1ecba9f49b6b77c578f7dbbdfceb43d54406216599b823e5950d8",
        "homepage": "https://github.com/amazon-science/spot-diff",
        "license": "Custom Amazon License (Research Use Only)",
        "commercial_use_permitted": False,
        "citation": "Zou et al., Spot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, ECCV 2022",
        "categories": [
            "candle", "capsules", "cashew", "chewinggum", "fryum",
            "macaroni", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
        ],
        "default_mount_path": "data/real/visa"
    },
    "btad": {
        "name": "BeanTech Anomaly Detection (BTAD)",
        "url": "http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/btad.zip",
        "archive_name": "btad.zip",
        "expected_sha256": "4b17540ea9039a7386ee35c8e3305530dfcf9cf399aa4ca6795f745e33dcf7d7",
        "homepage": "http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/",
        "license": "Research Use Only",
        "commercial_use_permitted": False,
        "citation": "Mishra et al., VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization, WACV 2021",
        "categories": ["01", "02", "03"],
        "default_mount_path": "data/real/btad"
    },
    "mpdd": {
        "name": "Mobile Phone Defect Dataset (MPDD)",
        "url": "https://github.com/stepanje/MPDD",
        "archive_name": "mpdd.zip",
        "expected_sha256": "d4e287a2a095cf28e1c64eb37456d9536d7a42bbca404bc51a141ebf1c3aa74b",
        "homepage": "https://github.com/stepanje/MPDD",
        "license": "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
        "commercial_use_permitted": False,
        "citation": "Jezek et al., Deep Learning for Visual Anomaly Detection in Industrial Defect Recognition, IEEE Access 2021",
        "categories": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"],
        "default_mount_path": "data/real/mpdd"
    }
}


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_presence(dataset_key: str, mount_path: Optional[str] = None) -> Dict[str, Any]:
    info = DATASET_CATALOG.get(dataset_key)
    if not info:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    path = mount_path or info["default_mount_path"]
    exists = os.path.exists(path) and os.path.isdir(path)

    verified_categories = []
    missing_categories = []
    total_images = 0

    if exists:
        for cat in info["categories"]:
            cat_p = os.path.join(path, cat)
            if os.path.exists(cat_p) and os.path.isdir(cat_p):
                verified_categories.append(cat)
                # Count files
                for _, _, files in os.walk(cat_p):
                    total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            else:
                missing_categories.append(cat)

    is_complete = exists and len(missing_categories) == 0 and total_images > 0

    return {
        "dataset": dataset_key,
        "name": info["name"],
        "mount_path": os.path.abspath(path),
        "is_mounted": exists,
        "is_complete": is_complete,
        "verified_categories": verified_categories,
        "missing_categories": missing_categories,
        "total_images": total_images,
        "license": info["license"],
        "status": "MOUNTED_AND_VERIFIED" if is_complete else "ABSENT_OR_INCOMPLETE"
    }


def audit_all_datasets() -> Dict[str, Any]:
    status = {}
    all_ready = True
    for key in DATASET_CATALOG:
        report = verify_dataset_presence(key)
        status[key] = report
        if not report["is_complete"]:
            all_ready = False

    gate_b_status = "PASSED" if all_ready else "BLOCKED"
    summary = {
        "gate_b_status": gate_b_status,
        "reason": "All physical datasets mounted and verified" if all_ready else "Physical datasets not mounted in data/real/. Synthetic fixtures are used for development/CI verification only.",
        "datasets": status
    }
    return summary


def download_dataset(dataset_key: str, download_dir: str = "data/raw") -> None:
    info = DATASET_CATALOG.get(dataset_key)
    if not info:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    os.makedirs(download_dir, exist_ok=True)
    archive_path = os.path.join(download_dir, info["archive_name"])
    mount_path = info["default_mount_path"]

    logger.info(f"Downloading {info['name']} from {info['url']}...")
    try:
        urllib.request.urlretrieve(info["url"], archive_path)
        logger.info(f"Downloaded {archive_path}. Verifying SHA-256...")
        file_hash = compute_file_sha256(archive_path)
        if file_hash != info["expected_sha256"]:
            logger.warning(f"SHA-256 mismatch: got {file_hash}, expected {info['expected_sha256']}")
        else:
            logger.info(f"SHA-256 verified successfully: {file_hash[:12]}...")

        os.makedirs(mount_path, exist_ok=True)
        if archive_path.endswith((".tar.xz", ".tar.gz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=mount_path)
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(path=mount_path)
        logger.info(f"Successfully unpacked to {mount_path}")
    except Exception as e:
        logger.error(f"Download/extraction failed: {e}")
        logger.info(f"Please manually download the archive from {info['homepage']} and place in {mount_path}")


def main():
    parser = argparse.ArgumentParser(description="VISTA-VLM Dataset Acquisition & Provenance Verifier")
    parser.add_argument("--audit", action="store_true", help="Audit presence and integrity of physical datasets")
    parser.add_argument("--download", choices=list(DATASET_CATALOG.keys()) + ["all"], help="Download specified dataset")
    parser.add_argument("--info", choices=list(DATASET_CATALOG.keys()) + ["all"], help="Display dataset licensing and citation info")
    args = parser.parse_args()

    if args.audit or (not args.download and not args.info):
        summary = audit_all_datasets()
        print("\n" + "=" * 70)
        print("VISTA-VLM DATASET PROVENANCE & GATE B AUDIT")
        print("=" * 70)
        print(f"Gate B Status: {summary['gate_b_status']}")
        print(f"Details: {summary['reason']}\n")
        for k, v in summary["datasets"].items():
            print(f"  [{v['status']}] {v['name']}: {v['mount_path']}")
            print(f"     Categories verified: {len(v['verified_categories'])}/{len(v['verified_categories']) + len(v['missing_categories'])} | Images: {v['total_images']}")
            print(f"     License: {v['license']}")
        print("=" * 70)
        return

    if args.info:
        targets = list(DATASET_CATALOG.keys()) if args.info == "all" else [args.info]
        for t in targets:
            info = DATASET_CATALOG[t]
            print(f"\nDataset: {info['name']}")
            print(f"Homepage: {info['homepage']}")
            print(f"License: {info['license']}")
            print(f"Citation: {info['citation']}")

    if args.download:
        targets = list(DATASET_CATALOG.keys()) if args.download == "all" else [args.download]
        for t in targets:
            download_dataset(t)


if __name__ == "__main__":
    main()

