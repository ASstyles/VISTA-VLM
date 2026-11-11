"""
VISTA-VLM MVTec AD Dataset Loader (V7.0 Research-Grade Upgrade).
================================================================
Implements strict four-partition scientific experiment protocol:
  - 'train': Nominal normal reference images (model parameter training & memory bank fitting)
  - 'val_calibration': Independent held-out normal images (split-conformal calibration)
  - 'val_tuning': Independent held-out labelled normal + defective images (selective prediction tuning & fusion optimization)
  - 'test': Final, untouched blind evaluation partition (normal + defective)

Enforces zero data leakage across all four partitions with deterministic assignment.
"""

from __future__ import annotations

import os
import glob
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
from vista_vlm.data.common_schema import ImageSample
from vista_vlm.data.base import BaseAnomalyDataset
from vista_vlm.utils.registry import DATASET_REGISTRY
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.data.mvtec")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


@DATASET_REGISTRY.register("mvtec")
class MVTecDataset(BaseAnomalyDataset):
    """
    MVTec Anomaly Detection (AD) Dataset Loader.
    Loads images, annotations, and pixel-level ground-truth masks.
    Enforces strict 4-way disjoint partitioning without leakage.
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
        split: str = "train",
        val_ratio: float = 0.20,
        tuning_ratio: float = 0.20,
        seed: int = 42,
        image_size: int = 518,
    ):
        self.category = category
        self.split = split
        self.val_ratio = val_ratio
        self.tuning_ratio = tuning_ratio
        self.seed = seed

        samples = self._load_samples(root_dir, category, val_ratio, tuning_ratio, seed)
        filtered = [s for s in samples if s.split == split]

        if not filtered and samples:
            # Backwards compatibility fallback if split requested is 'test' and tuning used
            if split == "test":
                filtered = [s for s in samples if s.split in ("test", "val_tuning")]

        super().__init__(filtered, image_size=image_size, is_training=(split == "train"))

    @staticmethod
    def _load_samples(
        root_dir: str,
        category: str,
        val_ratio: float = 0.20,
        tuning_ratio: float = 0.20,
        seed: int = 42
    ) -> List[ImageSample]:
        cat_dir = os.path.join(root_dir, category)
        if not os.path.exists(cat_dir):
            raise FileNotFoundError(f"MVTec category directory not found: {cat_dir}")

        samples: List[ImageSample] = []
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

        # 1. Train normal images -> deterministically split into 'train' and 'val_calibration'
        train_good_dir = os.path.join(cat_dir, "train", "good")
        if os.path.exists(train_good_dir):
            train_good_files = sorted(glob.glob(os.path.join(train_good_dir, "*.*")))
            train_good_files = [f for f in train_good_files if f.lower().endswith(valid_exts)]
            n_train_total = len(train_good_files)

            # Deterministic split: allocate at least 20 for val_calibration when n >= 38 to support alpha=0.05
            if n_train_total >= 38:
                n_val = 20
                val_files = set(train_good_files[-n_val:])
            elif n_train_total >= 4:
                n_val = max(1, int(n_train_total * val_ratio))
                val_files = set(train_good_files[-n_val:])
            elif n_train_total >= 2:
                val_files = {train_good_files[-1]}
            else:
                val_files = set(train_good_files)

            for p in train_good_files:
                with Image.open(p) as img:
                    w, h = img.size
                img_id = os.path.splitext(os.path.basename(p))[0]
                split_tag = "val_calibration" if p in val_files else "train"
                samples.append(ImageSample(
                    image_path=p,
                    mask_path=None,
                    category=category,
                    split=split_tag,
                    is_anomaly=False,
                    defect_type="good",
                    width=w,
                    height=h,
                    image_id=f"{category}_{split_tag}_{img_id}",
                    split_protocol="deterministic_4partition"
                ))

        # 2. Test images (good + defects) -> split into 'val_tuning' and 'test'
        test_dir = os.path.join(cat_dir, "test")
        gt_dir = os.path.join(cat_dir, "ground_truth")

        if os.path.exists(test_dir):
            defect_types = sorted(os.listdir(test_dir))
            for dtype in defect_types:
                dtype_path = os.path.join(test_dir, dtype)
                if not os.path.isdir(dtype_path):
                    continue

                img_files = sorted(glob.glob(os.path.join(dtype_path, "*.*")))
                img_files = [f for f in img_files if f.lower().endswith(valid_exts)]
                is_anomaly = (dtype != "good")
                n_files = len(img_files)

                # Partition test pool into val_tuning (e.g. 25%) and test (75%)
                if n_files >= 4 and tuning_ratio > 0.0:
                    n_tune = max(1, int(n_files * tuning_ratio))
                    tune_files = set(img_files[:n_tune])
                elif n_files >= 2 and tuning_ratio > 0.0:
                    tune_files = {img_files[0]}
                else:
                    tune_files = set()

                for p in img_files:
                    with Image.open(p) as img:
                        w, h = img.size
                    img_id = os.path.splitext(os.path.basename(p))[0]
                    mask_path = None

                    if is_anomaly:
                        expected_mask = os.path.join(gt_dir, dtype, f"{img_id}_mask.png")
                        if not os.path.exists(expected_mask):
                            expected_mask = os.path.join(gt_dir, dtype, f"{img_id}.png")
                        if os.path.exists(expected_mask):
                            mask_path = expected_mask

                    split_tag = "val_tuning" if p in tune_files else "test"
                    samples.append(ImageSample(
                        image_path=p,
                        mask_path=mask_path,
                        category=category,
                        split=split_tag,
                        is_anomaly=is_anomaly,
                        defect_type=dtype,
                        width=w,
                        height=h,
                        image_id=f"{category}_{split_tag}_{dtype}_{img_id}",
                        split_protocol="deterministic_4partition"
                    ))

        return samples
