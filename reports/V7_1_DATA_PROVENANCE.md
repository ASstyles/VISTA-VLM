# V7.1 Data Provenance & Release Gate B Audit Report

**Author**: VISTA-VLM Core Research Team  
**Branch**: `v7-research-validation-hardening`  
**Date**: November 2026  
**Artifact**: `reports/V7_1_DATA_PROVENANCE.md`  
**Audit Utility**: `scripts/acquire_datasets.py --audit`  

---

## 1. Executive Summary: Release Gate B Status

> [!CAUTION]
> **GATE B (REAL-WORLD DATASET QUALIFICATION): BLOCKED**  
> Physical real-world industrial inspection benchmark datasets (**MVTec AD**, **VisA**, **BTAD**, **MPDD**) are not mounted in `data/real/`.  
> All benchmark results reported in developer qualification environments derive from procedural synthetic fixtures (`data/synthetic_fixtures/`) designed exclusively for unit test and software pipeline verification.

In adherence to scientific integrity principles, VISTA-VLM V7.1 explicitly disallows representing developmental synthetic fixture evaluations as empirical real-world qualification. Release Gate B remains formally **BLOCKED** until physical benchmark datasets are downloaded, mounted, and verified.

---

## 2. Dataset Provenance & Catalog Audit

The table below documents the audit performed by `scripts/acquire_datasets.py`:

| Dataset | Expected Mount Location | Status | Total Verified Images | License | Commercial Permitted |
|:---|:---|:---:|:---:|:---|:---:|
| **MVTec AD** | `data/real/mvtec/` | **ABSENT** | 0 | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) | No |
| **VisA** | `data/real/visa/` | **ABSENT** | 0 | Custom Amazon License (Research Use Only) | No |
| **BTAD** | `data/real/btad/` | **ABSENT** | 0 | Academic Research Use Only | No |
| **MPDD** | `data/real/mpdd/` | **ABSENT** | 0 | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) | No |

---

## 3. Automated Dataset Acquisition Protocol

To enable seamless qualification when external network access or storage is provisioned, `scripts/acquire_datasets.py` provides:

### 3.1 Official Download Endpoints & Cryptographic Verification
- **MVTec AD**:
  - Source URL: `https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz`
  - Expected SHA-256: `4e3415c8f85f838634e0fe93c524e93144ef8795995cfbf0062bc6ec54db31f1`
- **VisA**:
  - Source URL: `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`
  - Expected SHA-256: `8d3ea51f08e1ecba9f49b6b77c578f7dbbdfceb43d54406216599b823e5950d8`
- **BTAD**:
  - Source URL: `http://dm.infor.uva.es/resources/beantech-anomaly-detection-dataset/btad.zip`
  - Expected SHA-256: `4b17540ea9039a7386ee35c8e3305530dfcf9cf399aa4ca6795f745e33dcf7d7`
- **MPDD**:
  - Source URL: `https://github.com/stepanje/MPDD`
  - Expected SHA-256: `d4e287a2a095cf28e1c64eb37456d9536d7a42bbca404bc51a141ebf1c3aa74b`

### 3.2 Acquisition Command Lines
```bash
# Print licensing and citation metadata
python scripts/acquire_datasets.py --info all

# Audit local physical mounting status
python scripts/acquire_datasets.py --audit

# Download and unpack official datasets (requires network access)
python scripts/acquire_datasets.py --download all
```

---

## 4. Synthetic Development Fixture Protocol

When physical assets are unmounted, developmental verification executes against procedural synthetic fixtures:
- **Location**: `data/synthetic_fixtures/{mvtec,visa}/`
- **Generation Script**: `scripts/prepare_benchmark_data.py`
- **Manifests**: `data/manifests/synthetic/`
- **Characteristics**: Multi-scale synthetic defects (40% hairline cracks/scratches, 40% geometric dents/chips, 20% organic Perlin stains) generated strictly on foreground object contours.
- **Manifest Tag**: All samples produced by this generator carry `split_protocol="procedural_synthetic_benchmark_fixture"`.

### Scientific Reporting Invariant
All generated qualification JSON artifacts and markdown tables MUST include:
```json
{
  "provenance": "procedural_synthetic_benchmark_fixture",
  "is_real": false,
  "guarantee_status": "SYNTHETIC_DEVELOPMENTAL_EVIDENCE_ONLY_NO_PRODUCTION_ESCAPE_GUARANTEE"
}
```
Representing these synthetic numbers as real MVTec or VisA benchmark scores is strictly prohibited.

