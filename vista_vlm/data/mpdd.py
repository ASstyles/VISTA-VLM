import os
import glob
from PIL import Image
from typing import List, Optional
from vista_vlm.data.common_schema import ImageSample
from vista_vlm.data.base import BaseAnomalyDataset
from vista_vlm.utils.registry import DATASET_REGISTRY

MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white",
    "connector", "metal_plate", "tubes"
]

@DATASET_REGISTRY.register("mpdd")
class MPDDDataset(BaseAnomalyDataset):
    """
    MPDD (Metal Parts Defect Dataset) Loader.
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "bracket_black",
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
            raise FileNotFoundError(f"MPDD category directory not found: {cat_dir}")

        samples = []
        rng = random.Random(seed)

        # Train normal
        train_good_dir = os.path.join(cat_dir, "train", "good")
        if os.path.exists(train_good_dir):
            train_files = sorted(glob.glob(os.path.join(train_good_dir, "*.*")))
            rng.shuffle(train_files)
            n_val = int(len(train_files) * val_ratio)
            val_files = set(train_files[:n_val])

            for p in train_files:
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
                    image_id=f"mpdd_{category}_{split_tag}_{img_id}"
                ))

        # Test
        test_dir = os.path.join(cat_dir, "test")
        gt_dir = os.path.join(cat_dir, "ground_truth")
        if os.path.exists(test_dir):
            for dtype in sorted(os.listdir(test_dir)):
                dtype_path = os.path.join(test_dir, dtype)
                if not os.path.isdir(dtype_path):
                    continue

                for p in sorted(glob.glob(os.path.join(dtype_path, "*.*"))):
                    with Image.open(p) as img:
                        w, h = img.size
                    img_id = os.path.splitext(os.path.basename(p))[0]
                    is_anomaly = (dtype != "good")
                    mask_path = None
                    if is_anomaly:
                        expected = os.path.join(gt_dir, dtype, f"{img_id}_mask.png")
                        if not os.path.exists(expected):
                            expected = os.path.join(gt_dir, dtype, f"{img_id}.png")
                        if os.path.exists(expected):
                            mask_path = expected

                    samples.append(ImageSample(
                        image_path=p,
                        mask_path=mask_path,
                        category=category,
                        split="test",
                        is_anomaly=is_anomaly,
                        defect_type=dtype,
                        width=w,
                        height=h,
                        image_id=f"mpdd_{category}_test_{dtype}_{img_id}"
                    ))

        return samples
