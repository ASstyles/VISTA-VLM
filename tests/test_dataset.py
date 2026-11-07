import pytest
import torch
import numpy as np
from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.visa import VisADataset
from vista_vlm.data.synthetic import SyntheticAnomalyGenerator, generate_perlin_noise_mask, generate_geometric_mask
from vista_vlm.data.transforms import VISTATransform

def test_mvtec_dataset_loading():
    train_ds = MVTecDataset("data/processed/mvtec", category="bottle", split="train", image_size=518)
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["image"].shape == (3, 518, 518)
    assert sample["mask"].shape == (1, 518, 518)
    assert sample["label"].item() == 0.0 # Train is normal only
    assert sample["category"] == "bottle"

def test_mvtec_test_split_masks():
    test_ds = MVTecDataset("data/processed/mvtec", category="bottle", split="test", image_size=518)
    assert len(test_ds) > 0
    # Check that defective samples have non-zero masks
    defective_found = False
    for i in range(len(test_ds)):
        sample = test_ds[i]
        if sample["label"].item() == 1.0:
            defective_found = True
            assert sample["mask"].max() > 0.0 # Has ground truth mask
            break
    assert defective_found

def test_few_shot_sampling():
    train_ds = MVTecDataset("data/processed/mvtec", category="bottle", split="train", image_size=518)
    few_ds = train_ds.sample_few_shot(k=4, seed=42)
    assert len(few_ds) == 4

def test_synthetic_anomaly_generator():
    gen = SyntheticAnomalyGenerator(anomaly_prob=1.0, seed=42)
    dummy_img = (np.random.rand(518, 518, 3) * 255).astype(np.uint8)
    anom_img, mask, is_anom = gen(dummy_img)
    assert is_anom is True
    assert anom_img.shape == (518, 518, 3)
    assert mask.shape == (518, 518)
    assert mask.max() == 1.0
    assert mask.min() == 0.0

def test_perlin_noise_mask():
    mask = generate_perlin_noise_mask(518, 518, max_scale=5, threshold=0.5)
    assert mask.shape == (518, 518)
    assert set(np.unique(mask)).issubset({0.0, 1.0})

def test_geometric_mask():
    mask = generate_geometric_mask(518, 518)
    assert mask.shape == (518, 518)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
