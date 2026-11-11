from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json
import os

@dataclass
class ImageSample:
    image_path: str
    mask_path: Optional[str]
    category: str
    split: str  # 'train', 'val_calibration', 'val_tuning', 'test'
    is_anomaly: bool
    defect_type: str  # 'good' for normal, or defect name like 'scratch', 'crack'
    width: int
    height: int
    image_id: str
    split_protocol: str = "default"  # 'official_split', 'deterministic_4partition', 'procedural'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageSample":
        return cls(**data)


@dataclass
class DatasetReport:
    dataset_name: str
    total_samples: int
    categories: List[str]
    split_distribution: Dict[str, int]
    anomaly_distribution: Dict[str, int]
    defect_types_per_category: Dict[str, List[str]]
    dimension_ranges: Dict[str, List[int]]
    mask_correspondence_valid: bool
    leakage_check_passed: bool

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
