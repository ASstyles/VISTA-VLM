import os
import pytest
import torch
import numpy as np
from PIL import Image
from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.visa import VisADataset
from vista_vlm.data.synthetic import (
    SyntheticAnomalyGenerator,
    generate_perlin_noise_mask,
    generate_geometric_mask,
    get_foreground_mask
)
from vista_vlm.data.transforms import VISTATransform
from vista_vlm.data.manifest_manager import DatasetIntegrityChecker, resolve_dataset_root


@pytest.fixture(scope="module")
def isolated_synthetic_root(tmp_path_factory):
    """Creates a temporary, strictly isolated synthetic dataset fixture for testing."""
    root = tmp_path_factory.mktemp("isolated_dataset")
    cat_dir = root / "bottle"
    train_dir = cat_dir / "train" / "good"
    test_good_dir = cat_dir / "test" / "good"
    test_anom_dir = cat_dir / "test" / "scratch"
    gt_dir = cat_dir / "ground_truth" / "scratch"

    train_dir.mkdir(parents=True)
    test_good_dir.mkdir(parents=True)
    test_anom_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    # 3 train images
    for i in range(3):
        arr = np.full((128, 128, 3), 50 + i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(str(train_dir / f"{i:03d}.png"))

    # 2 test normal
    for i in range(2):
        arr = np.full((128, 128, 3), 180 + i * 10, dtype=np.uint8)
        Image.fromarray(arr).save(str(test_good_dir / f"{i:03d}.png"))

    # 2 test anomaly with masks
    for i in range(2):
        arr = np.full((128, 128, 3), 120, dtype=np.uint8)
        arr[40:70, 40:70] = 255
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:70, 40:70] = 255
        Image.fromarray(arr).save(str(test_anom_dir / f"{i:03d}.png"))
        Image.fromarray(mask).save(str(gt_dir / f"{i:03d}_mask.png"))

    return str(root)


def test_mvtec_dataset_isolated_loading(isolated_synthetic_root):
    train_ds = MVTecDataset(isolated_synthetic_root, category="bottle", split="train", image_size=518)
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["image"].shape == (3, 518, 518)
    assert sample["mask"].shape == (1, 518, 518)
    assert sample["label"].item() == 0.0  # Train is normal only
    assert sample["category"] == "bottle"


def test_mvtec_test_split_masks(isolated_synthetic_root):
    test_ds = MVTecDataset(isolated_synthetic_root, category="bottle", split="test", image_size=518)
    assert len(test_ds) > 0
    defective_found = False
    for i in range(len(test_ds)):
        sample = test_ds[i]
        if sample["label"].item() == 1.0:
            defective_found = True
            assert sample["mask"].max() > 0.0
            break
    assert defective_found


def test_few_shot_sampling(isolated_synthetic_root):
    train_ds = MVTecDataset(isolated_synthetic_root, category="bottle", split="train", image_size=518)
    few_ds = train_ds.sample_few_shot(k=2, seed=42)
    assert len(few_ds) == min(2, len(train_ds))


def test_dataset_integrity_manifest(isolated_synthetic_root):
    manifest = DatasetIntegrityChecker.build_manifest(
        dataset_name="isolated_test_mvtec",
        root_dir=isolated_synthetic_root,
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    assert manifest["is_valid"] is True
    assert manifest["total_samples"] == 7
    assert len(manifest["integrity_errors"]) == 0
    assert manifest["samples"][0]["image_sha256"] != ""


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
    mask = generate_perlin_noise_mask(518, 518)
    assert mask.shape == (518, 518)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_geometric_mask():
    mask = generate_geometric_mask(518, 518)
    assert mask.shape == (518, 518)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_foreground_mask():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:80, 20:80] = 200
    fg = get_foreground_mask(dummy_img)
    assert fg.shape == (100, 100)
    assert fg[50, 50] > 0
