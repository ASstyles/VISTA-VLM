# VISTA-VLM V5.0 Comprehensive Application and Infrastructure Security Audit

**Document ID:** `V5-REP-SEC-001`  
**Version:** 5.0.0  
**Status:** AUDITED & REMEDIATED  
**Date:** 2026-11-11  
**Authors:** Application and Cloud Security Engineer & Independent Technical Auditor  
**Repository Branch:** `v5-real-world-qualification`  

---

## 1. Executive Summary

This security audit evaluated the VISTA-VLM inspection platform across the API surface, image ingestion pipeline, model serialization, and container infrastructure. 

Three high-severity vulnerabilities identified in V4.0 were remediated and verified with regression tests. All remaining findings have documented mitigation controls in `docs/SECURITY_OPERATIONS.md`.

### Vulnerability Summary Matrix
| Vulnerability ID | Category | Severity | Description | Remediation Status | Verification |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **SEC-V5-01** | Input Validation | **HIGH** | Arbitrary File Path Traversal in `POST /inspect/batch` | **RESOLVED** | `test_api_batch_path_traversal_rejection` |
| **SEC-V5-02** | Resource Management | **HIGH** | Memory Exhaustion (DoS) via Unbounded Uploads | **RESOLVED** | 20MB chunked streaming quota |
| **SEC-V5-03** | Ingestion Security | **MEDIUM** | Image Decompression Bomb (Pixel Flood Attack) | **RESOLVED** | Max 25M pixels, dimension bounds |
| **SEC-V5-04** | Deserialization | **MEDIUM** | Arbitrary Code Execution Risk via `torch.load` | **RESOLVED** | SHA-256 verification & `weights_only` |
| **SEC-V5-05** | API Security | **LOW** | Missing Authentication on Edge Telemetry Endpoints | **RESOLVED** | Bearer Token / API Key header |

---

## 2. In-Depth Vulnerability Analysis & Remediations

### 2.1 SEC-V5-01: Path Traversal in Batch Inspection (`POST /inspect/batch`)
- **Vulnerability**: In V4.0, the `batch` endpoint accepted raw client-supplied file paths (`image_paths: List[str]`) and called `os.path.exists()` and `Image.open()`. An attacker or compromised edge client could supply `../../etc/shadow` or `../../sensitive_data/` to probe internal server file structures.
- **Remediation**: Implemented strict canonical sandboxing via `validate_safe_path`:
  ```python
  def validate_safe_path(path: str, allowed_bases: Optional[List[str]] = None) -> Path:
      canonical_target = os.path.realpath(path)
      for base in allowed_bases:
          canonical_base = os.path.realpath(base)
          if os.path.commonpath([canonical_target, canonical_base]) == canonical_base:
              return Path(canonical_target)
      raise HTTPException(status_code=403, detail=f"Access denied: Path '{path}' outside allowed directories.")
  ```
- **Verification**: Tested against `../../etc/passwd` and absolute traversal paths; all invalid requests return HTTP 403 Forbidden.

### 2.2 SEC-V5-02: Denial of Service via Unbounded Uploads (`POST /inspect`)
- **Vulnerability**: In V4.0, `await file.read()` read entire incoming payloads into memory without size checks. An attacker uploading a 5GB stream would exhaust host RAM, crashing the inspection process and halting production.
- **Remediation**: Implemented streaming chunked ingestion with a hard quota of `MAX_IMAGE_BYTES = 20 * 1024 * 1024` (20MB):
  ```python
  MAX_IMAGE_BYTES = 20 * 1024 * 1024
  content = bytearray()
  while True:
      chunk = await file.read(64 * 1024)
      if not chunk:
          break
      content.extend(chunk)
      if len(content) > MAX_IMAGE_BYTES:
          raise HTTPException(status_code=413, detail="Uploaded image exceeds 20MB limit.")
  ```

### 2.3 SEC-V5-03: Image Decompression Bomb Defense
- **Vulnerability**: Specially crafted image headers (e.g., a 10KB PNG file declaring dimensions of $100,000 \times 100,000$) cause PIL to allocate over 30GB of memory during raster decompression.
- **Remediation**: Enforced global limits in `vista_vlm/api/routes.py`:
  1. `Image.MAX_IMAGE_PIXELS = 25_000_000` (caps maximum uncompressed pixel count).
  2. Executed `pil_img.verify()` to catch truncated or malformed image data streams.
  3. Re-opened and verified dimensions within permissible industrial bounds:
     $$\text{width}, \text{height} \in [16, 4096]$$
  4. Mandatory conversion to `RGB` format, automatically stripping malicious EXIF tags or embedded script payloads.

### 2.4 SEC-V5-04: Model Deserialization Integrity
- **Vulnerability**: Traditional PyTorch `.pth` files use Python `pickle`, which can execute arbitrary code if a malicious checkpoint replaces the production model.
- **Remediation**: 
  1. All production checkpoints must have their SHA-256 hash verified against an immutable configuration manifest before loading.
  2. PyTorch `weights_only=True` is enforced wherever supported.
  3. Safe-tensors migration path documented for future releases.

---

## 3. Container and Infrastructure Hardening

```dockerfile
# Production Docker Security Specification
FROM python:3.11-slim-bookworm

# Create non-privileged service user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/false appuser

WORKDIR /app
COPY --chown=appuser:appgroup . /app

# Run as non-root user
USER 10001:10001

# Read-only root filesystem with ephemeral /tmp mount
VOLUME ["/tmp"]
EXPOSE 8000

CMD ["uvicorn", "vista_vlm.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Security Configuration Checklist
- [x] Non-root execution (`UID 10001`).
- [x] Read-only root filesystem enabled in deployment manifests.
- [x] No plaintext credentials or secrets committed to repository.
- [x] CORS origins restricted to authorized dashboard endpoints.
- [x] Rate limiting configured on `/inspect` endpoints.

