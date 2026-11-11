"""
Unit tests for VISTA-VLM Dataset Loaders and Synthetic Fixture Integrity.
All tests use strictly isolated pytest temporary directories (tmp_path_factory).
No tests write to or depend on real dataset paths (data/real/).
"""

from __future__ import annotations

import os
import pytest
import numpy as np

from vista_vlm.data.mvtec import MVTecDataset
from vista_vlm.data.visa import VisADataset
from vista_vlm.data.synthetic import (
    SyntheticAnomalyGenerator,
    generate_perlin_noise_mask,
    generate_geometric_mask,
    get_foreground_mask
)
from vista_vlm.data.manifest_manager import DatasetIntegrityChecker
from scripts.prepare_benchmark_data import prepare_mvtec_fixtures, prepare_visa_fixtures


@pytest.fixture(scope="session")
def synthetic_dataset_roots(tmp_path_factory):
    """
    Session-level fixture that generates minimal procedural synthetic fixtures
    in an isolated temporary directory to prevent cross-test pollution.
    """
    root = tmp_path_factory.mktemp("vista-synthetic")
    mvtec_dir = str(root / "mvtec")
    visa_dir = str(root / "visa")
    real_guard = str(root / "real_guard")
    os.makedirs(real_guard, exist_ok=True)

    prepare_mvtec_fixtures(
        output_root=mvtec_dir,
        real_root=real_guard,
        num_train=3,
        num_test_good=2,
        num_test_anom=4,
        seed=100
    )
    prepare_visa_fixtures(
        output_root=visa_dir,
        real_root=real_guard,
        num_normal=5,
        num_anom=3,
        seed=200
    )

    return {
        "root": str(root),
        "mvtec": mvtec_dir,
        "visa": visa_dir,
    }


def test_mvtec_dataset_synthetic_loading(synthetic_dataset_roots):
    mvtec_root = synthetic_dataset_roots["mvtec"]
    train_ds = MVTecDataset(mvtec_root, category="bottle", split="train", image_size=518)
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["image"].shape == (3, 518, 518)
    assert sample["mask"].shape == (1, 518, 518)
    assert sample["label"].item() == 0.0  # Train is normal only
    assert sample["category"] == "bottle"


def test_mvtec_test_split_masks(synthetic_dataset_roots):
    mvtec_root = synthetic_dataset_roots["mvtec"]
    test_ds = MVTecDataset(mvtec_root, category="bottle", split="test", image_size=518)
    assert len(test_ds) > 0

    defective_count = 0
    normal_count = 0
    for i in range(len(test_ds)):
        sample = test_ds[i]
        if sample["label"].item() == 1.0:
            defective_count += 1
            assert sample["mask"].max() > 0.0, "Defective sample must have non-zero mask"
        else:
            normal_count += 1
            assert sample["mask"].max() == 0.0, "Normal sample mask must be all zeros"

    assert defective_count > 0, "Test split must contain defective samples"
    assert normal_count > 0, "Test split must contain normal samples"


def test_few_shot_sampling(synthetic_dataset_roots):
    mvtec_root = synthetic_dataset_roots["mvtec"]
    train_ds = MVTecDataset(mvtec_root, category="bottle", split="train", image_size=518)
    few_ds = train_ds.sample_few_shot(k=2, seed=42)
    assert len(few_ds) == min(2, len(train_ds))


def test_visa_dataset_synthetic_loading(synthetic_dataset_roots):
    visa_root = synthetic_dataset_roots["visa"]
    train_ds = VisADataset(visa_root, category="candle", split="train", image_size=518)
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["image"].shape == (3, 518, 518)
    assert sample["label"].item() == 0.0
    assert sample["category"] == "candle"


def test_dataset_integrity_manifest(synthetic_dataset_roots):
    mvtec_root = synthetic_dataset_roots["mvtec"]
    manifest = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_mvtec_test",
        root_dir=mvtec_root,
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    assert manifest["is_valid"] is True
    assert manifest["provenance"] == "procedural_synthetic_benchmark_fixture"
    assert manifest["total_samples"] > 0
    assert len(manifest["integrity_errors"]) == 0
    assert manifest["samples"][0]["image_sha256"] != ""


def test_missing_dataset_raises_file_not_found(tmp_path):
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        MVTecDataset(empty_dir, category="nonexistent_part", split="train")


def test_malformed_dataset_rejected(tmp_path):
    malformed_dir = tmp_path / "malformed"
    cat_dir = malformed_dir / "bottle"
    cat_dir.mkdir(parents=True)
    # Missing train/good folder completely
    manifest = DatasetIntegrityChecker.build_manifest(
        dataset_name="malformed_test",
        root_dir=str(malformed_dir),
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    # Manifest will find 0 samples, hence is_valid is False
    assert manifest["is_valid"] is False


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
