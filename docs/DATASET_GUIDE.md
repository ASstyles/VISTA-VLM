# VISTA-VLM V5.0 Dataset Engineering and Provenance Guide

**Document ID:** `V5-DOC-DAT-001`  
**Version:** 5.0.0  
**Status:** APPROVED  
**Date:** 2026-11-11  

---

## 1. Segregated Data Architecture

In compliance with **Non-Negotiable Principle 1.2 (Strict Separation of Evidence)**, VISTA-VLM V5.0 enforces permanent filesystem separation:

```
data/
  synthetic_fixtures/         # Procedural test fixtures (CI & local unit tests only)
    mvtec_like/               # Conforming to MVTec AD schema (seed-randomized)
    visa_like/                # Conforming to VisA schema

  real/                       # Genuine industrial datasets (External mount / authorized access)
    mvtec_ad/                 # 15 official MVTec AD categories (CC BY-NC-SA 4.0)
    visa/                     # 12 official VisA categories (CC BY 4.0)
    btad/                     # BeanTech Anomaly Detection (3 categories)
    mpdd/                     # Metal Parts Defect Dataset (6 categories)
    real_iad/                 # Real-IAD multi-view dataset

  manifests/                  # Cryptographic SHA-256 dataset manifests
    synthetic/                # mvtec_like_manifest.json, visa_like_manifest.json
    real/                     # Generated on archive unpack
```

**Rule**: Synthetic procedural fixtures must **never** be written to or read from `data/real/`.

---

## 2. Common Data Schema

All datasets are normalized into a unified schema (`vista_vlm.data.common_schema.ImageSample`):

```python
class ImageSample(BaseModel):
    sample_id: str
    image_path: str
    mask_path: Optional[str] = None
    category: str
    dataset_name: str
    split: str             # 'train', 'val_calibration', 'test'
    is_anomaly: bool       # False for nominal, True for defective
    defect_type: str       # 'good' for nominal, or specific defect category
    image_width: int
    image_height: int
```

---

## 3. Cryptographic Manifest and Leakage Verification Protocol

Before training or benchmarking, datasets must be audited with `vista_vlm.data.manifest_manager`:

```bash
# Generate and audit dataset manifest:
python -c "
from vista_vlm.data.manifest_manager import generate_dataset_manifest
manifest = generate_dataset_manifest(
    data_dir='data/synthetic_fixtures/mvtec_like',
    output_path='data/manifests/synthetic/mvtec_like_manifest.json',
    dataset_name='mvtec_like',
    version='5.0.0'
)
print('Manifest valid:', manifest['is_valid'])
"
```

The manifest audit enforces:
1. **Zero Data Leakage**: Computes SHA-256 hashes of all images; flags an error if any test image is identical or near-duplicate to a train image.
2. **Mask Correspondence**: Verifies that every defective test image has a corresponding ground-truth mask with identical $(W, H)$ dimensions.
3. **Directory Conformance**: Verifies required subfolders (`train/good`, `test/good`, `test/<defect>`, `ground_truth/<defect>`).

---

## 4. Configuring Real Dataset Paths

To evaluate against mounted real datasets:
```bash
export VISTA_DATA_ROOT=/path/to/authorized/datasets
python scripts/run_multi_track_benchmark.py
```
If `VISTA_DATA_ROOT` is unset, the loader falls back cleanly to `data/synthetic_fixtures` and logs a transparent disclosure that synthetic benchmark fixtures are being used.
