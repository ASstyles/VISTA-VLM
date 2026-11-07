import os
import glob
from PIL import Image
from typing import List, Optional
from vista_vlm.data.common_schema import ImageSample
from vista_vlm.data.base import BaseAnomalyDataset
from vista_vlm.utils.registry import DATASET_REGISTRY

BTAD_CATEGORIES = ["01", "02", "03"]

@DATASET_REGISTRY.register("btad")
class BTADDataset(BaseAnomalyDataset):
    """
    BeanTech Anomaly Detection (BTAD) Dataset Loader.
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "01",
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
            raise FileNotFoundError(f"BTAD category directory not found: {cat_dir}")

        samples = []
        rng = random.Random(seed)

        # Train ok images
        train_ok_dir = os.path.join(cat_dir, "train", "ok")
        if os.path.exists(train_ok_dir):
            train_files = sorted(glob.glob(os.path.join(train_ok_dir, "*.*")))
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
                    defect_type="ok",
                    width=w,
                    height=h,
                    image_id=f"btad_{category}_{split_tag}_{img_id}"
                ))

        # Test ok and ko images
        test_ok_dir = os.path.join(cat_dir, "test", "ok")
        if os.path.exists(test_ok_dir):
            for p in sorted(glob.glob(os.path.join(test_ok_dir, "*.*"))):
                with Image.open(p) as img:
                    w, h = img.size
                img_id = os.path.splitext(os.path.basename(p))[0]
                samples.append(ImageSample(
                    image_path=p,
                    mask_path=None,
                    category=category,
                    split="test",
                    is_anomaly=False,
                    defect_type="ok",
                    width=w,
                    height=h,
                    image_id=f"btad_{category}_test_ok_{img_id}"
                ))

        test_ko_dir = os.path.join(cat_dir, "test", "ko")
        gt_dir = os.path.join(cat_dir, "ground_truth", "ko")
        if os.path.exists(test_ko_dir):
            for p in sorted(glob.glob(os.path.join(test_ko_dir, "*.*"))):
                with Image.open(p) as img:
                    w, h = img.size
                img_id = os.path.splitext(os.path.basename(p))[0]
                mask_path = os.path.join(gt_dir, f"{img_id}.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(gt_dir, f"{img_id}.bmp")
                samples.append(ImageSample(
                    image_path=p,
                    mask_path=mask_path if os.path.exists(mask_path) else None,
                    category=category,
                    split="test",
                    is_anomaly=True,
                    defect_type="ko",
                    width=w,
                    height=h,
                    image_id=f"btad_{category}_test_ko_{img_id}"
                ))

        return samples
