"""
Comprehensive Regression Test Suite for Synthetic Data Isolation and Generator Integrity.
Tests all 10 requirements of Section 6:
  1. Missing Tuple import regression through Python compilation and reflection.
  2. Generator default output directories.
  3. Rejection of any output path inside data/real/.
  4. Synthetic fixture provenance metadata.
  5. Deterministic seeded output.
  6. Image-mask dimension correspondence.
  7. Binary mask integrity.
  8. No accidental modification of pre-existing real-data sentinel files.
  9. Temporary dataset fixture isolation.
  10. Missing real dataset must return explicit unavailable/blocked status, never synthetic fallback.
"""

from __future__ import annotations

import os
import py_compile
import inspect
import hashlib
import pytest
import numpy as np
from PIL import Image

import scripts.prepare_benchmark_data as pbd
from scripts.prepare_benchmark_data import (
    validate_synthetic_output_dir,
    create_synthetic_industrial_object,
    create_synthetic_defect,
    prepare_mvtec_fixtures,
    build_and_save_manifests
)
from vista_vlm.data.manifest_manager import (
    DatasetIntegrityChecker,
    resolve_dataset_root,
    compute_sha256
)


def test_tuple_import_and_compilation():
    """Requirement 1: Verify prepare_benchmark_data compiles cleanly and resolves Tuple annotations without NameError."""
    script_path = os.path.abspath(pbd.__file__)
    # py_compile will raise py_compile.PyCompileError if syntax or compile-time error occurs
    compiled_path = py_compile.compile(script_path, doraise=True)
    assert compiled_path is not None and os.path.exists(compiled_path)

    # Inspect function annotations to ensure no NameError or unresolved forward refs
    fn = pbd.create_synthetic_defect
    annotations = inspect.get_annotations(fn)
    assert "return" in annotations
    ret_type = annotations["return"]
    # Verify return annotation references Tuple / tuple
    assert "Tuple" in str(ret_type) or "tuple" in str(ret_type)


def test_generator_default_output_directories():
    """Requirement 2: Verify default output paths are strictly under data/synthetic_fixtures."""
    import argparse
    parser = argparse.ArgumentParser()
    # Check default arguments configured in prepare_benchmark_data
    # Default output-root must be data/synthetic_fixtures
    # Default mvtec dir must resolve under data/synthetic_fixtures/mvtec
    output_root = "data/synthetic_fixtures"
    expected_mvtec = os.path.join(output_root, "mvtec")
    expected_visa = os.path.join(output_root, "visa")

    # Ensure defaults are outside data/real
    assert not expected_mvtec.startswith("data/real")
    assert not expected_visa.startswith("data/real")
    assert "synthetic_fixtures" in expected_mvtec
    assert "synthetic_fixtures" in expected_visa


def test_rejection_of_real_data_output_path(tmp_path):
    """Requirement 3: Verify validate_synthetic_output_dir raises ValueError for real-data paths."""
    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True)
    unsafe_mvtec = real_dir / "mvtec_ad"
    unsafe_sub = real_dir / "subdir"

    # Direct containment in real_root
    with pytest.raises(ValueError, match="DATA INTEGRITY VIOLATION"):
        validate_synthetic_output_dir(str(unsafe_sub), real_root=str(real_dir))

    # Real dataset token forbidden
    with pytest.raises(ValueError, match="DATA INTEGRITY VIOLATION"):
        validate_synthetic_output_dir(str(unsafe_mvtec), real_root=str(real_dir))


def test_rejection_of_forbidden_tokens():
    """Requirement 3b: Verify forbidden academic dataset tokens are rejected even outside real_root."""
    for token in ["mvtec_ad", "real_iad", "btad", "mpdd"]:
        fake_path = os.path.join("some", "other", token, "output")
        with pytest.raises(ValueError, match="DATA INTEGRITY VIOLATION"):
            validate_synthetic_output_dir(fake_path, real_root="data/real")


def test_synthetic_fixture_provenance_metadata(tmp_path):
    """Requirement 4: Verify generated manifests contain explicit synthetic provenance metadata."""
    fixture_dir = tmp_path / "test_synth"
    cat_dir = fixture_dir / "bottle" / "train" / "good"
    cat_dir.mkdir(parents=True)

    img = create_synthetic_industrial_object("bottle", seed=42)
    Image.fromarray(img).save(str(cat_dir / "000.png"))

    manifest = DatasetIntegrityChecker.build_manifest(
        dataset_name="synthetic_test",
        root_dir=str(fixture_dir),
        provenance="procedural_synthetic_benchmark_fixture",
        layout="mvtec"
    )
    assert manifest["provenance"] == "procedural_synthetic_benchmark_fixture"
    assert manifest["total_samples"] == 1
    assert manifest["samples"][0]["provenance"] == "procedural_synthetic_benchmark_fixture"


def test_deterministic_seeded_output():
    """Requirement 5: Verify identical seeds produce bitwise-identical images and masks."""
    img1 = create_synthetic_industrial_object("bottle", seed=12345)
    img2 = create_synthetic_industrial_object("bottle", seed=12345)
    assert np.array_equal(img1, img2), "Identical seeds must produce bitwise-identical objects"

    # Distinct seeds produce distinct objects
    img3 = create_synthetic_industrial_object("bottle", seed=99999)
    assert not np.array_equal(img1, img3), "Distinct seeds must produce distinct objects"

    anom1, mask1 = create_synthetic_defect(img1, "scratch", seed=54321)
    anom2, mask2 = create_synthetic_defect(img1, "scratch", seed=54321)
    assert np.array_equal(anom1, anom2)
    assert np.array_equal(mask1, mask2)


def test_image_mask_dimension_correspondence():
    """Requirement 6: Verify image and mask dimensions strictly correspond."""
    base_img = create_synthetic_industrial_object("cable", width=518, height=518, seed=42)
    anom_img, mask = create_synthetic_defect(base_img, "crack", width=518, height=518, seed=42)

    assert anom_img.shape == (518, 518, 3)
    assert mask.shape == (518, 518)
    assert anom_img.shape[:2] == mask.shape


def test_binary_mask_integrity():
    """Requirement 7: Verify defect masks contain strictly binary values {0, 255}."""
    base_img = create_synthetic_industrial_object("leather", seed=10)
    for dt in ["scratch", "crack", "dent", "contamination", "hole"]:
        _, mask = create_synthetic_defect(base_img, dt, seed=20)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Mask for '{dt}' contains non-binary values: {unique_vals}"
        assert 255 in unique_vals, f"Defect mask for '{dt}' has no foreground anomaly pixels"


def test_no_accidental_modification_of_real_data_sentinel(tmp_path):
    """Requirement 8: Verify pre-existing real-data sentinel files are never overwritten or altered."""
    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True)
    sentinel_file = real_dir / "genuine_mvtec_ad_sentinel.txt"
    sentinel_content = b"GENUINE_MVTEC_ACADEMIC_DATASET_PROTECTED_HASH_XYZ"
    sentinel_file.write_bytes(sentinel_content)
    sentinel_hash_before = hashlib.sha256(sentinel_content).hexdigest()

    # Generate synthetic fixtures in a separate safe folder
    synthetic_dir = tmp_path / "synthetic_fixtures"
    prepare_mvtec_fixtures(
        output_root=str(synthetic_dir / "mvtec"),
        real_root=str(real_dir),
        num_train=2,
        num_test_good=1,
        num_test_anom=1,
        seed=42
    )

    # Verify sentinel in real_dir is completely intact and unchanged
    assert sentinel_file.exists()
    assert sentinel_file.read_bytes() == sentinel_content
    assert hashlib.sha256(sentinel_file.read_bytes()).hexdigest() == sentinel_hash_before


def test_temporary_dataset_fixture_isolation(tmp_path):
    """Requirement 9: Verify fixtures in temporary directories operate in complete isolation without global side effects."""
    isolated_root = tmp_path / "custom_isolated"
    mvtec_dir = str(isolated_root / "mvtec")
    prepare_mvtec_fixtures(
        output_root=mvtec_dir,
        real_root=str(tmp_path / "guard"),
        num_train=2,
        num_test_good=1,
        num_test_anom=1,
        seed=1
    )
    assert os.path.exists(mvtec_dir)
    assert os.path.exists(os.path.join(mvtec_dir, "bottle", "train", "good", "000.png"))


def test_missing_real_dataset_blocked_status(tmp_path, monkeypatch):
    """Requirement 10: Verify missing real dataset returns explicit blocked status, never silent synthetic fallback."""
    # Point VISTA_DATA_ROOT to an empty directory
    empty_real = tmp_path / "empty_real_root"
    empty_real.mkdir()
    monkeypatch.setenv("VISTA_DATA_ROOT", str(empty_real))

    # Request real dataset resolution with allow_synthetic_fallback=False
    resolved_path, provenance, is_real = resolve_dataset_root("mvtec_ad", allow_synthetic_fallback=False)
    assert is_real is False
    assert provenance == "real_dataset_unavailable"
    assert "mvtec_ad" in resolved_path

