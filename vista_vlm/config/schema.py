from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import yaml
import os

class SystemConfig(BaseModel):
    seed: int = 42
    device: str = "cpu"
    mixed_precision: bool = False
    num_workers: int = 0

class DataConfig(BaseModel):
    root_dir: str = "data"
    dataset_name: str = "mvtec"
    category: str = "bottle"
    categories: Optional[List[str]] = None
    image_size: int = 518
    patch_size: int = 14
    grid_size: int = 37
    normal_only_train: bool = True
    train_batch_size: int = 4
    eval_batch_size: int = 4
    few_shot_k: Optional[int] = None

class SyntheticConfig(BaseModel):
    enabled: bool = True
    perlin_scale: int = 6
    perlin_min_scale: int = 0
    noise_threshold: float = 0.5
    opacity_range: Tuple[float, float] = (0.2, 0.95)
    geometric_prob: float = 0.5
    texture_perturb_prob: float = 0.5

class FusionConfig(BaseModel):
    type: str = "global_local"
    proj_dim: int = 256
    global_dim: int = 384
    use_film_conditioning: bool = True

class DecoderConfig(BaseModel):
    type: str = "progressive_conv"
    in_channels: int = 256
    hidden_dims: List[int] = [128, 64, 32, 16]
    out_channels: int = 1
    dropout: float = 0.1

class ScoringConfig(BaseModel):
    top_k_ratio: float = 0.01
    global_head_weight: float = 0.5

class ModelConfig(BaseModel):
    backbone: str = "dinov2_vits14"
    embed_dim: int = 384
    out_indices: List[int] = [4, 8, 12]
    freeze_backbone: bool = True
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

class TrainingConfig(BaseModel):
    epochs: int = 15
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    lambda_dice: float = 1.0
    lambda_image: float = 0.5
    save_dir: str = "checkpoints"
    val_interval: int = 1

class LanguageVerifierConfig(BaseModel):
    enabled: bool = True
    clip_model: str = "ViT-B/16"
    embed_dim: int = 512
    roi_threshold: float = 0.5
    min_box_area: int = 100
    box_padding: float = 0.15
    ambiguity_margin: float = 0.10
    min_confidence: float = 0.25

class OODConfig(BaseModel):
    enabled: bool = True
    metric: str = "mahalanobis"
    threshold_percentile: float = 98.0

class SelectiveConfig(BaseModel):
    uncertainty_band_width: float = 0.08
    enable_human_escalation: bool = True

class CalibrationConfig(BaseModel):
    method: str = "split_conformal"
    target_false_alarm_rate: float = 0.05
    calibration_split_ratio: float = 0.20
    ood: OODConfig = Field(default_factory=OODConfig)
    selective: SelectiveConfig = Field(default_factory=SelectiveConfig)

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

class VISTAConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    language_verifier: LanguageVerifierConfig = Field(default_factory=LanguageVerifierConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def load_from_yaml(cls, path: str) -> "VISTAConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
