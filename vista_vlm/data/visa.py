import os
import glob
from PIL import Image
from typing import List, Optional
from vista_vlm.data.common_schema import ImageSample
from vista_vlm.data.base import BaseAnomalyDataset
from vista_vlm.utils.registry import DATASET_REGISTRY

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

@DATASET_REGISTRY.register("visa")
class VisADataset(BaseAnomalyDataset):
    """
    VisA (Visual Anomaly) Dataset Loader.
    Standardized to the common schema.
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "candle",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        image_size: int = 518,
    ):
        samples = self._load_samples(root_dir, category, val_ratio, seed)
        filtered = [s for s in samples if s.split == split]
        super().__init__(filtered, image_size=image_size, is_training=(split == "train"))
        self.category = category
        self.split = split

    @staticmethod
    def _load_samples(root_dir: str, category: str, val_ratio: float = 0.2, seed: int = 42) -> List[ImageSample]:
        import random
        cat_dir = os.path.join(root_dir, category)
        if not os.path.exists(cat_dir):
            raise FileNotFoundError(f"VisA category directory not found: {cat_dir}")

        samples = []
        rng = random.Random(seed)

        # Standard VisA layout: Data/Images/Normal, Data/Images/Anomaly, Data/Masks/Anomaly
        # Or split-based layout: train/normal, test/normal, test/anomaly
        normal_dirs = [
            os.path.join(cat_dir, "Data", "Images", "Normal"),
            os.path.join(cat_dir, "train", "normal"),
            os.path.join(cat_dir, "normal")
        ]
        anomaly_dirs = [
            os.path.join(cat_dir, "Data", "Images", "Anomaly"),
            os.path.join(cat_dir, "test", "anomaly"),
            os.path.join(cat_dir, "anomaly")
        ]
        mask_dirs = [
            os.path.join(cat_dir, "Data", "Masks", "Anomaly"),
            os.path.join(cat_dir, "ground_truth"),
            os.path.join(cat_dir, "mask")
        ]

        found_normal = [d for d in normal_dirs if os.path.exists(d)]
        found_anomaly = [d for d in anomaly_dirs if os.path.exists(d)]
        found_mask = [d for d in mask_dirs if os.path.exists(d)]

        if not found_normal:
            raise FileNotFoundError(f"No normal images directory found for VisA category {category}")

        normal_files = sorted(glob.glob(os.path.join(found_normal[0], "*.*")))
        rng.shuffle(normal_files)
        
        # Split normal images into train (70%), val_calibration (15%), test normal (15%)
        n_total = len(normal_files)
        n_val = int(n_total * val_ratio)
        n_test_normal = int(n_total * 0.15)
        
        val_files = set(normal_files[:n_val])
        test_normal_files = set(normal_files[n_val:n_val + n_test_normal])

        for p in normal_files:
            with Image.open(p) as img:
                w, h = img.size
            img_id = os.path.splitext(os.path.basename(p))[0]
            if p in val_files:
                split_tag = "val_calibration"
            elif p in test_normal_files:
                split_tag = "test"
            else:
                split_tag = "train"

            samples.append(ImageSample(
                image_path=p,
                mask_path=None,
                category=category,
                split=split_tag,
                is_anomaly=False,
                defect_type="normal",
                width=w,
                height=h,
                image_id=f"visa_{category}_{split_tag}_{img_id}"
            ))

        # Anomalous test images
        if found_anomaly:
            anomaly_files = sorted(glob.glob(os.path.join(found_anomaly[0], "*.*")))
            mask_dir = found_mask[0] if found_mask else None
            for p in anomaly_files:
                with Image.open(p) as img:
                    w, h = img.size
                img_id = os.path.splitext(os.path.basename(p))[0]
                mask_path = None
                if mask_dir:
                    expected = os.path.join(mask_dir, f"{img_id}.png")
                    if os.path.exists(expected):
                        mask_path = expected

                samples.append(ImageSample(
                    image_path=p,
                    mask_path=mask_path,
                    category=category,
                    split="test",
                    is_anomaly=True,
                    defect_type="anomaly",
                    width=w,
                    height=h,
                    image_id=f"visa_{category}_test_anomaly_{img_id}"
                ))

        return samples
