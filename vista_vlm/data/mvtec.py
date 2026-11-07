import os
import glob
from PIL import Image
from typing import List, Optional
from vista_vlm.data.common_schema import ImageSample
from vista_vlm.data.base import BaseAnomalyDataset
from vista_vlm.utils.registry import DATASET_REGISTRY

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
    Separates train (normal only), val_calibration (held-out normal), and test (normal + defective).
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
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
            raise FileNotFoundError(f"MVTec category directory not found: {cat_dir}")

        samples = []
        rng = random.Random(seed)

        # 1. Train normal images -> split into 'train' and 'val_calibration'
        train_good_dir = os.path.join(cat_dir, "train", "good")
        if os.path.exists(train_good_dir):
            train_good_files = sorted(glob.glob(os.path.join(train_good_dir, "*.*")))
            rng.shuffle(train_good_files)
            n_val = int(len(train_good_files) * val_ratio)
            val_files = set(train_good_files[:n_val])

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
                    image_id=f"{category}_{split_tag}_{img_id}"
                ))

        # 2. Test images (good + defects)
        test_dir = os.path.join(cat_dir, "test")
        gt_dir = os.path.join(cat_dir, "ground_truth")

        if os.path.exists(test_dir):
            defect_types = sorted(os.listdir(test_dir))
            for dtype in defect_types:
                dtype_path = os.path.join(test_dir, dtype)
                if not os.path.isdir(dtype_path):
                    continue

                img_files = sorted(glob.glob(os.path.join(dtype_path, "*.*")))
                for p in img_files:
                    with Image.open(p) as img:
                        w, h = img.size
                    img_id = os.path.splitext(os.path.basename(p))[0]
                    is_anomaly = (dtype != "good")
                    mask_path = None

                    if is_anomaly:
                        # Find matching mask in ground_truth/{dtype}/{img_id}_mask.png
                        expected_mask = os.path.join(gt_dir, dtype, f"{img_id}_mask.png")
                        if not os.path.exists(expected_mask):
                            # Try without _mask suffix
                            expected_mask = os.path.join(gt_dir, dtype, f"{img_id}.png")
                        if os.path.exists(expected_mask):
                            mask_path = expected_mask

                    samples.append(ImageSample(
                        image_path=p,
                        mask_path=mask_path,
                        category=category,
                        split="test",
                        is_anomaly=is_anomaly,
                        defect_type=dtype,
                        width=w,
                        height=h,
                        image_id=f"{category}_test_{dtype}_{img_id}"
                    ))

        return samples
