"""
Packaging and Distribution Regression Tests for VISTA-VLM.
Validates:
  1. Package installation without sys.path modification
  2. Package discovery by setuptools
  3. Metadata consistency between pyproject.toml and __version__
  4. Importability of all top-level subpackages and foundational classes
"""

import os
import sys
import glob
import zipfile
import re
import importlib
import importlib.metadata
import pytest
from setuptools import find_packages
from packaging.version import Version


def _load_pyproject_version() -> str:
    """Parses project version directly from pyproject.toml."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    pyproject_path = os.path.abspath(pyproject_path)
    if not os.path.exists(pyproject_path):
        pytest.skip(f"pyproject.toml not found at {pyproject_path}")
    try:
        import tomllib
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return str(data["project"]["version"])
    except ImportError:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
        raise ValueError(f"Could not parse version from {pyproject_path}")


def test_package_import():
    """Confirms vista_vlm is installed in site-packages / editable mode."""
    import vista_vlm
    assert hasattr(vista_vlm, "__file__"), "vista_vlm has no __file__ attribute"
    assert vista_vlm.__file__ is not None
    assert "vista_vlm" in sys.modules


def test_package_version_consistency():
    """Ensures __version__, pyproject.toml, and installed distribution metadata all match exactly."""
    import vista_vlm

    expected_version = Version(_load_pyproject_version())
    code_version = Version(vista_vlm.__version__)
    assert code_version == expected_version, (
        f"vista_vlm.__version__ ({vista_vlm.__version__}) does not match pyproject.toml ({expected_version})"
    )

    try:
        dist_raw = importlib.metadata.version("vista-vlm")
        dist_version = Version(dist_raw)
        assert dist_version == expected_version, (
            f"Installed distribution version ({dist_raw}) does not match pyproject.toml ({expected_version})"
        )
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vista-vlm package metadata not registered in current environment")


def test_package_metadata():
    """Validates installed package metadata via standard importlib with zero stale allowlists."""
    expected_version = Version(_load_pyproject_version())

    try:
        meta = importlib.metadata.metadata("vista-vlm")
        assert meta["Name"] == "vista-vlm"
        installed_version = Version(meta["Version"])
        assert installed_version == expected_version, (
            f"Metadata Version ({meta['Version']}) does not match pyproject.toml ({expected_version})"
        )
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


def test_wheel_metadata_matches_project_version():
    """If wheels exist in dist/, inspects WHEEL/METADATA to ensure version consistency."""
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "dist")
    if not os.path.exists(dist_dir):
        pytest.skip("No dist/ directory found")
    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    if not wheels:
        pytest.skip("No wheel built yet in dist/")

    expected_version = Version(_load_pyproject_version())
    for whl in wheels:
        with zipfile.ZipFile(whl, "r") as z:
            meta_filenames = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(meta_filenames) == 1, f"Expected exactly 1 METADATA file in {whl}"
            with z.open(meta_filenames[0]) as mf:
                content = mf.read().decode("utf-8")
                match = re.search(r"^Version:\s*([^\r\n]+)", content, re.MULTILINE)
                assert match is not None, f"No Version header in {whl}"
                whl_version = Version(match.group(1).strip())
                assert whl_version == expected_version, (
                    f"Wheel {whl} version ({whl_version}) does not match pyproject.toml ({expected_version})"
                )

