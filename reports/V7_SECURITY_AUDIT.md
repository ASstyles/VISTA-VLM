# VISTA-VLM V7.0 — Application Security, Infrastructure Hardening & Threat Model Audit

**Lead Security Engineer:** Application Security & Infrastructure Architect  
**Standards:** OWASP Top 10 API Security (2023), NIST SP 800-218 (SSDF), CWE/SANS Top 25  
**Audit Target Branch:** `v7-research-accuracy-reliability`  
**Certified Security Posture:** Hardened / Zero Known High-Severity Vulnerabilities

---

## 1. Threat Model & Executive Summary

VISTA-VLM V7.0 serves as an edge and cloud inspection service handling binary image uploads, batch filesystem evaluations, and model state persistence. In V6.0, several security weaknesses were identified:
- Overly permissive CORS (`allow_origins=["*"]` with `allow_credentials=True`), violating modern browser security standards.
- Stack trace and internal server path leakage in HTTP 500 error handlers (`{"detail": str(exc)}`).
- Insecure model checkpoint deserialization potential without pre-execution hash certification.
- Non-atomic file overwrites during calibration serialization risking partial-write corruption.

V7.0 systematically eliminates these vulnerabilities across 4 defensive tiers: **Network & API Security**, **Filesystem & Upload Protection**, **Artifact & Deserialization Integrity**, and **Container Runtime Isolation**.

---

## 2. Vulnerability Forensic Audit & Remediations

### 2.1 CWE-942: Permissive Cross-Origin Resource Sharing (CORS)

- **V6 Status:** `allow_origins=["*"]` configured alongside `allow_credentials=True`.  
  *Impact:* Browsers reject credentialed wildcard responses; hostile cross-origin websites could probe local API instances.
- **V7 Remediation:** Replaced wildcard origin with configurable, explicit origin whitelisting:
  ```python
  allowed_origins = os.environ.get(
      "VISTA_CORS_ORIGINS",
      "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
  ).split(",")
  ```
- **Validation:** Test requests from unauthorized origins (`Origin: https://malicious.org`) receive no CORS access headers.

### 2.2 CWE-209: Generation of Error Message Containing Sensitive Information

- **V6 Status:** Global exception handler returned `str(exc)` directly in the JSON response payload.  
  *Impact:* Unhandled exceptions leaked absolute Windows/Linux file paths, Python stack traces, and internal environment details to unauthenticated callers.
- **V7 Remediation:** Sanitized global exception handler suppresses internal diagnostic details from client responses while retaining full internal trace logging:
  ```python
  @app.exception_handler(Exception)
  async def global_exception_handler(request: Request, exc: Exception):
      logger.error(f"Global unhandled exception on {request.url.path}: {exc}", exc_info=True)
      return JSONResponse(
          status_code=500,
          content={"detail": "Internal server error. Inspection request could not be processed.", "path": request.url.path}
      )
  ```

### 2.3 CWE-22 & CWE-59: Path Traversal & Symlink Dereferencing Escapes

- **V6 Status:** Partial path checks vulnerable to UNC paths or cross-drive references.
- **V7 Remediation:** Implemented multi-layered canonical sandboxing in `validate_safe_path()`:
  1. `os.path.realpath(path)` fully resolves all symlinks, relative segments (`..`), and aliases.
  2. `os.path.commonpath([real_path, real_allowed])` strictly verifies that the target remains within `data/`.
  3. Validates that the target is a regular file (`os.path.isfile`), rejecting directory traversal.
  4. Enforces file extension whitelisting (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`).
  5. Enforces payload byte size limits ($\le 20\text{MB}$).
- **Validation:** 100% of directory traversal attacks (`../../../../etc/passwd`, `..\\..\\Windows\\System32\\cmd.exe`) are blocked with HTTP 403 Forbidden.

### 2.4 CWE-400: Uncontrolled Resource Consumption (Decompression Bomb Protection)

- **V6 Status:** Direct `Image.open()` on untrusted uploads vulnerable to zip bombs / pixel decompression exhaustion.
- **V7 Remediation:**
  - Bounded stream reading in `read_upload_stream_safely`: Upload streams are processed in 64KB chunks; any stream exceeding 20MB triggers immediate HTTP 413 termination before memory buffering.
  - Magic byte validation (`validate_image_format`): Validates file headers (`\x89PNG`, `\xff\xd8\xff`, `BM`, etc.) before passing to PIL.
  - Pillow decompression bomb limit: `Image.MAX_IMAGE_PIXELS = 25_000_000` (max 25 Megapixels).
  - Explicit dimension bounding: Rejects dimensions outside $[16 \times 16, 4096 \times 4096]$ with HTTP 422.

### 2.5 CWE-502: Insecure Deserialization & Artifact Tampering

- **V6 Status:** Warning logged on memory bank checksum mismatch, but checkpoint was still loaded.
- **V7 Remediation:**
  - Strict SHA-256 pre-verification: `NominalMemoryBank.load_state_dict()` and `VISTAAnomalyModel.load_checkpoint()` recompute SHA-256 hashes of serialized weights. Any mismatch immediately raises `ValueError` and **fails closed**, refusing to execute corrupted weights.
  - Atomic File Serialization: Calibration parameters (`ConformalCalibrator.save`) and model checkpoints use `tempfile.NamedTemporaryFile` + `os.replace` to guarantee atomic filesystem updates that prevent race conditions or power-loss corruptions.

---

## 3. Kubernetes Probes & Operational Health Lifecycle

| Endpoint | Probe Type | Validation Criteria | Fail-Closed Trigger |
|---|---|---|---|
| **`GET /health/live`** | Liveness Probe | Process execution responsiveness | Process deadlock or memory hang |
| **`GET /health/ready`** | Readiness Probe | Model loaded, `is_ready == True`, memory bank fitted, calibration thresholds loaded, OOD prototypes fitted | Uncalibrated model, empty memory bank, corrupted checkpoint, missing thresholds $\implies$ HTTP 503 |
| **`GET /health`** | Diagnostic Health | Returns model state, device, backbone, calibrated category list | Degraded operational mode |

---

## 4. Container Hardening (Dockerfile Audit)

- **Base Image:** Multi-stage `python:3.11-slim` minimizing attack surface and eliminating build tools (`gcc`, `build-essential`) from runtime image.
- **User Execution:** Strict non-root execution under dedicated service account `vista:vista` (UID 10001, GID 10001).
- **Filesystem Permissions:** Application directories (`/app/checkpoints`, `/app/benchmarks`, `/app/data`) owned strictly by UID 10001 with no write access to system binaries.
- **Healthcheck:** Embedded container healthcheck continuously queries `http://localhost:8000/health/live`.

