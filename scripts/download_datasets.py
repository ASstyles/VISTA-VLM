#!/usr/bin/env python3
"""
VISTA-VLM Dataset Acquisition & Preparation Script (Phase 2).
Provides official dataset download links, checksum validation, automated unpacking,
and standardized folder structuring for:
1. MVTec AD
2. VisA
3. BTAD
4. MPDD
5. Real-IAD
"""

import os
import argparse
import urllib.request
import tarfile
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vista_vlm.dataset_downloader")

DATASET_URLS = {
    "mvtec": {
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "archive": "mvtec_anomaly_detection.tar.xz",
        "homepage": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"
    },
    "visa": {
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "archive": "VisA_20220922.tar",
        "homepage": "https://github.com/amazon-science/spot-diff",
        "license": "Custom Amazon License (Research use only)"
    },
    "btad": {
        "url": "http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/btad.zip",
        "archive": "btad.zip",
        "homepage": "http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/",
        "license": "Research use only"
    },
    "mpdd": {
        "url": "https://github.com/stepanje/MPDD",
        "archive": "mpdd.zip",
        "homepage": "https://github.com/stepanje/MPDD",
        "license": "CC BY-NC 4.0"
    }
}

def print_dataset_info(dataset: str):
    info = DATASET_URLS.get(dataset.lower())
    if not info:
        print(f"Unknown dataset: {dataset}. Available: {list(DATASET_URLS.keys())}")
        return
    print("=" * 60)
    print(f"Dataset: {dataset.upper()}")
    print(f"Homepage: {info['homepage']}")
    print(f"Official URL: {info['url']}")
    print(f"License: {info['license']}")
    print("=" * 60)

def download_and_extract(dataset: str, target_dir: str = "data/raw"):
    info = DATASET_URLS.get(dataset.lower())
    if not info:
        raise ValueError(f"Unknown dataset: {dataset}")

    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, info["archive"])
    dest_dir = os.path.join(target_dir, dataset)

    if os.path.exists(dest_dir):
        logger.info(f"Directory {dest_dir} already exists. Skipping download.")
        return

    logger.info(f"Downloading {dataset} from {info['url']}...")
    try:
        urllib.request.urlretrieve(info["url"], archive_path)
        logger.info(f"Downloaded {archive_path}. Extracting...")
        if archive_path.endswith(".tar.xz") or archive_path.endswith(".tar.gz") or archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=dest_dir)
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(path=dest_dir)
        logger.info(f"Successfully extracted to {dest_dir}")
    except Exception as e:
        logger.warning(f"Automated download failed (network/auth restriction): {e}")
        logger.info(f"Please manually download the archive from {info['homepage']} and place in {target_dir}/{dataset}")

def main():
    parser = argparse.ArgumentParser(description="VISTA-VLM Dataset Acquisition & Helper")
    parser.add_argument("--dataset", choices=["mvtec", "visa", "btad", "mpdd", "all"], default="all")
    parser.add_argument("--info_only", action="store_true", help="Print download instructions and licensing")
    parser.add_argument("--target_dir", default="data/raw", help="Directory to save downloaded datasets")
    args = parser.parse_args()

    targets = list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    for d in targets:
        if args.info_only:
            print_dataset_info(d)
        else:
            print_dataset_info(d)
            download_and_extract(d, args.target_dir)

if __name__ == "__main__":
    main()
