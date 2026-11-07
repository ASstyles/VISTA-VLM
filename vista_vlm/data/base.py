import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from vista_vlm.data.common_schema import ImageSample, DatasetReport
from vista_vlm.data.transforms import VISTATransform

class BaseAnomalyDataset(Dataset):
    """
    Abstract base dataset for industrial anomaly detection and localization.
    Supports normal-only training, few-shot sampling, calibration split, and test evaluation.
    """
    def __init__(
        self,
        samples: List[ImageSample],
        transform: Optional[VISTATransform] = None,
        image_size: int = 518,
        is_training: bool = False,
    ):
        self.samples = samples
        self.transform = transform or VISTATransform(image_size=image_size, is_training=is_training)
        self.image_size = image_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        if not os.path.exists(sample.image_path):
            raise FileNotFoundError(f"Image file does not exist: {sample.image_path}")
        image = Image.open(sample.image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load mask if defective
        mask = None
        if sample.is_anomaly and sample.mask_path and os.path.exists(sample.mask_path):
            mask = Image.open(sample.mask_path).convert("L")
        elif sample.is_anomaly and sample.mask_path:
            # If specified mask file is missing, warn or raise
            raise FileNotFoundError(f"Ground truth mask file missing for defective sample: {sample.mask_path}")

        # Apply transformation
        img_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(1.0 if sample.is_anomaly else 0.0, dtype=torch.float32),
            "defect_type": sample.defect_type,
            "category": sample.category,
            "image_id": sample.image_id,
            "orig_size": (orig_w, orig_h),
            "image_path": sample.image_path,
        }

    def filter_split(self, split: str) -> "BaseAnomalyDataset":
        filtered = [s for s in self.samples if s.split == split]
        return BaseAnomalyDataset(filtered, transform=self.transform, image_size=self.image_size, is_training=(split == "train"))

    def sample_few_shot(self, k: int, seed: int = 42) -> "BaseAnomalyDataset":
        """Deterministic few-normal-shot subset sampling."""
        import random
        rng = random.Random(seed)
        normal_samples = [s for s in self.samples if not s.is_anomaly]
        if len(normal_samples) < k:
            selected = normal_samples
        else:
            selected = rng.sample(normal_samples, k)
        return BaseAnomalyDataset(selected, transform=self.transform, image_size=self.image_size, is_training=self.is_training)

    def generate_report(self, dataset_name: str) -> DatasetReport:
        categories = sorted(list(set(s.category for s in self.samples)))
        splits = {}
        for s in self.samples:
            splits[s.split] = splits.get(s.split, 0) + 1
        
        anomalies = {"normal": 0, "anomalous": 0}
        defect_types = {}
        dim_ranges = {"width": [float("inf"), 0], "height": [float("inf"), 0]}
        
        valid_masks = True
        for s in self.samples:
            if s.is_anomaly:
                anomalies["anomalous"] += 1
                if not (s.mask_path and os.path.exists(s.mask_path)):
                    valid_masks = False
            else:
                anomalies["normal"] += 1

            defect_types.setdefault(s.category, set()).add(s.defect_type)
            dim_ranges["width"][0] = min(dim_ranges["width"][0], s.width)
            dim_ranges["width"][1] = max(dim_ranges["width"][1], s.width)
            dim_ranges["height"][0] = min(dim_ranges["height"][0], s.height)
            dim_ranges["height"][1] = max(dim_ranges["height"][1], s.height)

        # Leakage check: ensure no test image path exists in train or val_calibration
        train_paths = set(s.image_path for s in self.samples if s.split == "train")
        val_paths = set(s.image_path for s in self.samples if s.split == "val_calibration")
        test_paths = set(s.image_path for s in self.samples if s.split == "test")
        leakage = bool(train_paths.intersection(test_paths) or val_paths.intersection(test_paths))

        return DatasetReport(
            dataset_name=dataset_name,
            total_samples=len(self.samples),
            categories=categories,
            split_distribution=splits,
            anomaly_distribution=anomalies,
            defect_types_per_category={k: sorted(list(v)) for k, v in defect_types.items()},
            dimension_ranges=dim_ranges,
            mask_correspondence_valid=valid_masks,
            leakage_check_passed=not leakage,
        )
