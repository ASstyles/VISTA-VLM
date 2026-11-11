#!/usr/bin/env python3
"""
VISTA-VLM Environment Validation Script
Detects system architecture, CPU/GPU capabilities, memory, PyTorch setup,
and package dependencies.
"""

import sys
import os
import platform
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vista_vlm.env_validator")

def validate_environment() -> dict:
    info = {
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "torch": {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "cudnn_enabled": False,
        },
        "packages": {},
        "hardware_recommendation": {}
    }

    # Test PyTorch
    try:
        import torch
        info["torch"]["installed"] = True
        info["torch"]["version"] = torch.__version__
        info["torch"]["cuda_available"] = torch.cuda.is_available()
        info["torch"]["device_count"] = torch.cuda.device_count()
        info["torch"]["cudnn_enabled"] = torch.backends.cudnn.enabled if hasattr(torch.backends, "cudnn") else False

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                info["torch"]["devices"].append({
                    "id": i,
                    "name": prop.name,
                    "total_memory_gb": round(prop.total_memory / (1024**3), 2),
                    "major": prop.major,
                    "minor": prop.minor,
                })
        else:
            info["torch"]["devices"].append({
                "id": -1,
                "name": "CPU (Optimized with multi-threading)",
                "threads": torch.get_num_threads(),
            })
    except ImportError as e:
        logger.warning(f"PyTorch not installed: {e}")

    # Test required dependencies
    required = [
        "torchvision", "numpy", "scipy", "sklearn", "PIL",
        "cv2", "matplotlib", "pydantic", "fastapi", "uvicorn", "yaml"
    ]
    for pkg in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            info["packages"][pkg] = {"status": "ok", "version": ver}
        except ImportError:
            info["packages"][pkg] = {"status": "missing", "version": None}

    # Determine recommended execution device and batch size
    if info["torch"]["cuda_available"]:
        info["hardware_recommendation"] = {
            "device": "cuda",
            "batch_size": 16,
            "mixed_precision": True,
            "num_workers": 2,
        }
    else:
        info["hardware_recommendation"] = {
            "device": "cpu",
            "batch_size": 4,
            "mixed_precision": False,
            "num_workers": 0,
        }

    return info

def main():
    info = validate_environment()
    print("=" * 60)
    print("VISTA-VLM ENVIRONMENT DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"OS: {info['os']['platform']} ({info['os']['architecture']})")
    print(f"Python: {info['python']['version'].split()[0]} ({info['python']['executable']})")
    print(f"PyTorch Version: {info['torch']['version']}")
    print(f"CUDA Available: {info['torch']['cuda_available']}")
    print(f"Target Execution Device: {info['hardware_recommendation']['device']}")
    print(f"Recommended Batch Size: {info['hardware_recommendation']['batch_size']}")
    print("-" * 60)
    print("Core Package Status:")
    all_ok = True
    for pkg, status in info["packages"].items():
        st = status["status"]
        ver = status["version"]
        print(f"  - {pkg:<15}: {st.upper():<8} (version: {ver})")
        if st != "ok":
            all_ok = False
    print("=" * 60)

    # Save diagnostic to file
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "environment_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Diagnostic saved to {out_path}")

    return 0 if all_ok and info["torch"]["installed"] else 1

if __name__ == "__main__":
    sys.exit(main())
