# VISTA-VLM Dataset Acquisition and Engineering Guide

## 1. Supported Benchmark Datasets

VISTA-VLM provides standardized loaders, parsers, and validation reports for four primary industrial anomaly datasets:

1. **MVTec AD (Anomaly Detection)**:
   - 15 industrial categories (5 textures, 10 objects): `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.
   - Resolution: 700×700 to 1024×1024 RGB.
   - Annotations: Exact pixel-level binary ground truth masks for all test defect types.
   - Licensing: CC BY-NC-SA 4.0.

2. **VisA (Visual Anomaly)**:
   - 12 complex multi-instance categories: `capsules`, `candle`, `macaroni`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, `chewinggum`, `fryum`, `pipe_fryum`, `cashews`, `print`.
   - High spatial complexity with multiple foreground instances.
   - Licensing: CC BY 4.0.

3. **BTAD (BeanTech Anomaly Detection)**:
   - 3 industrial manufacturing surface categories (`01`, `02`, `03`).
   - Body-centered industrial component scans.

4. **MPDD (Metal Parts Defect Dataset)**:
   - Variable background illumination and perspective shifts.

---

## 2. Common Annotation Schema

All datasets are normalized into a single unified schema (`vista_vlm.data.common_schema.ImageSample`):

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

## 3. Data Split & Zero-Leakage Protocol

To ensure rigorous scientific validity:
1. **Train Split**: Pure nominal/flawless images only ($y = 0$). No real defects are ever observed during feature learning or decoder training.
2. **Calibration Split**: Independent held-out normal images ($y = 0$) used exclusively for split conformal quantile threshold calculation and OOD covariance fitting.
3. **Test Split**: Contains both nominal images and all real defect classes with corresponding pixel ground truth masks.
4. **Synthetic Anomalies**: Generated exclusively on-the-fly from nominal training images via CutPaste, Perlin-blended perturbations, and geometric polygons. No test defect geometries or masks are referenced during synthesis.

---

## 4. Benchmark Data Preparation & Verification

Generate standardized dataset splits and validation reports:
```bash
python scripts/prepare_benchmark_data.py
```
This produces `benchmarks/dataset_validation_report.json` verifying:
- Image-mask dimension correspondence
- Binary mask value integrity (`{0, 255}`)
- Total sample counts per category
- Strict zero-leakage split partitions.

---

## 5. Benchmark Provenance & Transparency (SEC-15)

The repository provides lightweight, procedural synthetic benchmark fixtures conforming strictly to official MVTec AD and VisA directory schemas (`train/good`, `val_calibration/good`, `test/good`, `test/defect_name`, `ground_truth/defect_name`).
- **Tag**: `"dataset_provenance": "procedural_synthetic_benchmark_fixture"`
- **Purpose**: Enables immediate, deterministic, reproducible testing and regression audits on CI/CD pipelines and local CPU/GPU development environments without requiring multi-gigabyte external tarball downloads.
- **Production Conversion**: When benchmarking on physical raw industrial archives, simply replace the contents of `data/processed/mvtec/` and `data/processed/visa/` with extracted official archives; all loaders and evaluation scripts operate without modification.
