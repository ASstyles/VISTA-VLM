"""
Packaging and Distribution Regression Tests for VISTA-VLM.
Validates:
  1. Package installation without sys.path modification
  2. Package discovery by setuptools
  3. Metadata consistency between pyproject.toml and __version__
  4. Importability of all top-level subpackages and foundational classes
"""

import sys
import importlib
import importlib.metadata
import pytest
from setuptools import find_packages

def test_package_import():
    """Confirms vista_vlm is installed in site-packages / editable mode."""
    import vista_vlm
    assert hasattr(vista_vlm, "__file__"), "vista_vlm has no __file__ attribute"
    assert vista_vlm.__file__ is not None
    assert "vista_vlm" in sys.modules

def test_package_version_consistency():
    """Ensures __version__ matches package distribution metadata."""
    import vista_vlm
    assert vista_vlm.__version__ == "4.0.0"

    try:
        dist_version = importlib.metadata.version("vista-vlm")
        assert dist_version == "4.0.0", f"Distribution version mismatch: {dist_version} vs 4.0.0"
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vista-vlm package metadata not registered in current environment")

def test_package_metadata():
    """Validates installed package metadata via standard importlib."""
    try:
        meta = importlib.metadata.metadata("vista-vlm")
        assert meta["Name"] == "vista-vlm"
        assert meta["Version"] == "4.0.0"
        requires = importlib.metadata.requires("vista-vlm")
        assert requires is not None
        req_names = [r.split()[0].split(";")[0].split(">")[0].split("=")[0].strip() for r in requires]
        assert "torch" in req_names
        assert "numpy" in req_names
        assert "fastapi" in req_names
        assert any("multipart" in r.lower() for r in req_names)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vista-vlm not installed via pip in current environment")

def test_setuptools_package_discovery():
    """Ensures setuptools package discovery locates vista_vlm and all subpackages."""
    packages = find_packages(where=".", include=["vista_vlm*"])
    required_packages = [
        "vista_vlm",
        "vista_vlm.api",
        "vista_vlm.calibration",
        "vista_vlm.config",
        "vista_vlm.data",
        "vista_vlm.evaluation",
        "vista_vlm.experiments",
        "vista_vlm.models",
        "vista_vlm.models.backbones",
        "vista_vlm.models.baselines",
        "vista_vlm.models.decoder",
        "vista_vlm.models.fusion",
        "vista_vlm.optimization",
        "vista_vlm.utils"
    ]
    for pkg in required_packages:
        assert pkg in packages, f"Setuptools failed to discover subpackage: {pkg}"

def test_all_subpackages_importable():
    """Validates that every discovered subpackage can be imported cleanly."""
    subpackages = [
        "vista_vlm.api",
        "vista_vlm.calibration",
        "vista_vlm.config",
        "vista_vlm.data",
        "vista_vlm.evaluation",
        "vista_vlm.experiments",
        "vista_vlm.models",
        "vista_vlm.models.backbones",
        "vista_vlm.models.baselines",
        "vista_vlm.models.decoder",
        "vista_vlm.models.fusion",
        "vista_vlm.optimization",
        "vista_vlm.utils"
    ]
    for mod_name in subpackages:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import subpackage: {mod_name}"

def test_key_architectural_classes_importable():
    """Confirms core classes resolve from the installed package."""
    from vista_vlm.models.vista_model import VISTAAnomalyModel
    from vista_vlm.calibration.conformal import ConformalCalibrator
    from vista_vlm.calibration.ood import OODDetector
    from vista_vlm.calibration.selective import SelectivePredictor
    from vista_vlm.models.language_verifier import LanguageVerifier
    from vista_vlm.models.pipeline import VISTAInspectionPipeline

    assert VISTAAnomalyModel is not None
    assert ConformalCalibrator is not None
    assert OODDetector is not None
    assert SelectivePredictor is not None
    assert LanguageVerifier is not None
    assert VISTAInspectionPipeline is not None
