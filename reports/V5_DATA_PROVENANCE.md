# VISTA-VLM v5.0 — Data Provenance, Licensing, and Dataset Integrity Report

**Authority:** Independent Technical Auditor & Dataset Reliability Engineering  
**Date:** November 2026  
**System Version:** `5.0.0`  
**Classification:** Scientific & Legal Provenance Audit  

---

## 1. Non-Negotiable Data Integrity Policy

Under VISTA-VLM v5.0 governance:
1. **Zero Fabrication:** Procedural synthetic fixtures are never presented as real industrial datasets.
2. **Strict Segregation:** Synthetic engineering fixtures (`data/synthetic_fixtures/`) and real datasets (`data/real/`) are permanently separated in the filesystem, in code loaders, and in benchmark reporting.
3. **Cryptographic Traceability:** Every sample evaluated by VISTA-VLM must possess an immutable SHA-256 digest, category tag, split identifier, defect type, and provenance tag.
4. **Licensing Respect:** Academic research licenses (e.g., CC BY-NC-SA 4.0) are explicitly distinguished from commercial manufacturing deployment rights. Restricted dataset archives are never committed to version control.

---

## 2. Public Industrial Research Datasets Review

The following table documents the authoritative public industrial inspection datasets supported by VISTA-VLM:

| Dataset | Official Source / Homepage | License | Categories | Objects vs Textures | Annotation Format | Current Workstation Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **MVTec AD** | [MVTec Software GmbH](https://www.mvtec.com/company/research/datasets/mvtec-ad) | CC BY-NC-SA 4.0 (Non-Commercial Research Only) | 15 (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper) | 10 objects, 5 textures | High-resolution RGB ($700\times700$ to $1024\times1024$), 1-bit PNG defect masks | **BLOCKED** (External Download Pending) |
| **VisA** | [Amazon Science (Spot-the-Difference)](https://github.com/amazon-science/spot-diff) | Custom Amazon License (Non-Commercial Research Only) | 12 (candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum) | Complex multi-part assemblies, printed circuit boards | High-resolution RGB ($960\times960$ to $1500\times1000$), 1-bit PNG defect masks | **BLOCKED** (External Download Pending) |
| **BTAD** | [BeanTech Anomaly Detection](http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/) | Research Use Only | 3 (01, 02, 03 - industrial manufacturing components) | Industrial metal & plastic parts | High-resolution RGB ($1600\times1600$ to $2048\times2048$), PNG defect masks | **BLOCKED** (External Download Pending) |
| **MPDD** | [Metrology-Proved Defect Dataset](https://github.com/stepanje/MPDD) | CC BY-NC 4.0 | 6 (bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes) | Painted metallic sheet metal & plastic brackets | Variable perspectives, non-uniform illumination, PNG masks | **BLOCKED** (External Download Pending) |
| **MVTec AD 2** | [MVTec Software GmbH (2024)](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Proprietary Academic Agreement | Extended industrial components with multi-camera views | Industrial assemblies with 3D point cloud & multi-spectral data | High-res RGB + 3D point clouds + 2D masks | **BLOCKED** (External Download Pending) |
| **Real-IAD** | [Real-World Industrial Anomaly Detection](https://github.com/m-bain/real-iad) | Research Use Only | 30 categories | Metal, wood, plastic, fabric, multi-view setups | Multi-angle views, pixel-level masks | **BLOCKED** (External Download Pending) |

---

## 3. Workstation Dataset Availability & Acquisition State

### 3.1 Local Workstation Audit
- **Network Environment:** Isolated terminal sandbox execution without external public internet access (`[Errno 11001] getaddrinfo failed`).
- **Environment Variable `VISTA_DATA_ROOT`:** Not defined (`None`).
- **Local Directory `data/real/`:** Established with standardized subdirectories (`mvtec_ad/`, `visa/`, `btad/`, `mpdd/`, `real_iad/`), currently empty pending local authorized mounting.
- **Local Directory `data/synthetic_fixtures/`:** Fully populated with procedural fixtures (`mvtec_like`, `visa_like`) and verified cryptographic SHA-256 manifests (`data/manifests/synthetic/`).

### 3.2 Formal Operational Status
Pursuant to **Non-Negotiable Principle 1.4**, the status of empirical benchmarks on real public datasets is:
$$\mathbf{STATUS: \quad BLOCKED \quad (EXTERNAL \quad ASSET \quad PENDING)}$$

**Reason:** The raw physical image archives for MVTec AD (~5.2 GB), VisA (~4.1 GB), and BTAD (~1.8 GB) have not been mounted to this isolated workstation environment, and network access is restricted in standard sandbox mode.

---

## 4. Dataset Ingestion and Cryptographic Verification Protocol

When an authorized researcher or site engineer supplies the datasets, the following verifiable protocol must be executed:

### Step 1: Mount or Download
Place uncompressed or extracted archives in `data/real/<dataset_name>` or configure:
```bash
export VISTA_DATA_ROOT=/mnt/datasets/industrial_anomaly
```

### Step 2: Run Cryptographic Manifest Generator
```bash
python -c "from vista_vlm.data.manifest_manager import DatasetIntegrityChecker; \
manifest = DatasetIntegrityChecker.build_manifest('mvtec_ad', 'data/real/mvtec_ad', provenance='real_public_dataset', layout='mvtec'); \
DatasetIntegrityChecker.save_manifest(manifest, 'data/manifests/real/mvtec_ad_manifest.json'); \
print('Verified samples:', manifest['total_samples'], 'Valid:', manifest['is_valid'])"
```

### Step 3: Automated Integrity Verification
The `DatasetIntegrityChecker` executes:
1. **Image File Integrity:** Verifies decodability with PIL and OpenCV.
2. **Dimension Consistency:** Ensures image $(W, H)$ matches ground-truth mask $(W, H)$ exactly.
3. **Data Leakage Check:** Computes pairwise SHA-256 digests across `train` and `test` splits. Fails immediately if any training image appears in the test set.
4. **Mask Correspondence:** Verifies that every sample marked `is_anomaly=True` has a corresponding non-empty ground truth binary mask.
5. **Class Imbalance Telemetry:** Logs ratio of nominal vs defective samples per category.

---

## 5. Synthetic Fixture Disclosure

Procedural synthetic fixtures generated by `scripts/prepare_benchmark_data.py`:
- **Directory:** `data/synthetic_fixtures/mvtec_like/` and `data/synthetic_fixtures/visa_like/`
- **Samples Generated:** 235 MVTec-like samples across 5 categories; 135 VisA-like samples across 3 categories.
- **Manifest:** `data/manifests/synthetic/mvtec_like_manifest.json` and `visa_like_manifest.json`.
- **Integrity Validation:** 100% passed (0 data leakage errors, 0 missing masks, 0 dimensional mismatches).
- **Intended Purpose:** Strictly for continuous integration, regression testing, API input validation, and unit test execution.
- **Explicit Restriction:** Results derived from these fixtures are tagged with `"dataset_provenance": "procedural_synthetic_benchmark_fixture"` and must never be cited as empirical industrial defect detection rates.

