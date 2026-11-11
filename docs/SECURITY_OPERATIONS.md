# VISTA-VLM V5.0 Security Operations and Hardening Runbook

**Document ID:** `V5-DOC-SEC-001`  
**Version:** 5.0.0  
**Target:** DevSecOps & Production Systems Engineers  
**Classification:** SECURITY PROTOCOL & OPERATIONAL GUIDE  

---

## 1. Security Architecture & Threat Surface

VISTA-VLM operates as an edge or cloud API server ingesting binary image payloads from industrial cameras and automated sorting cells. The primary threat vectors are:
1. **Malicious Ingestion**: Uploading specially crafted image streams to cause denial of service (DoS) or arbitrary code execution.
2. **File Path Traversal**: Supplying arbitrary local paths to exfiltrate host credentials or probe internal files.
3. **Model Tampering**: Replacing trusted PyTorch checkpoints with poisoned or trojaned weight files.

---

## 2. Hardened Ingestion Defenses

### 2.1 File Path Sandboxing (`POST /inspect/batch`)
Raw file path inspection is strictly quarantined to white-listed filesystem directories. The `validate_safe_path` utility calculates the canonical real path using `os.path.realpath` and validates containment via `os.path.commonpath`.
- **Policy**: Any path attempting directory traversal (`../` or absolute symlinks outside permitted storage roots) is rejected immediately with HTTP 403 Forbidden.

### 2.2 Upload Streaming Quota (`POST /inspect`)
- **Quota Limit**: `MAX_IMAGE_BYTES = 20 * 1024 * 1024` (20 Megabytes).
- **Ingestion Mode**: Chunked streaming in 64 Kilobyte buffers. Payloads exceeding 20MB are terminated mid-stream and return HTTP 413 Payload Too Large, preventing memory exhaustion attacks.

### 2.3 Image Decompression Bomb Protection
- **Pixel Limit**: `Image.MAX_IMAGE_PIXELS = 25_000_000` (caps raster decompression at 25 Megapixels).
- **Stream Verification**: PIL `pil_img.verify()` is executed to certify valid image container syntax before decompression.
- **Dimension Bounds**: Validated strictly within $[\text{width}, \text{height}] \in [16, 4096]$.
- **Metadata Sanitization**: All images are re-rasterized into pure RGB numpy arrays, permanently stripping embedded EXIF tags, GPS metadata, and malicious payload scripts.

---

## 3. Model Serialization and Checkpoint Integrity

1. **SHA-256 Digest Verification**:
   Before loading any PyTorch `.pth` checkpoint into memory, the loading module calculates the file's SHA-256 hash and compares it against the expected manifest hash. Mismatches abort loading immediately with `ModelIntegrityError`.
2. **Safe Deserialization (`weights_only=True`)**:
   Where supported by PyTorch, checkpoints are loaded using `torch.load(..., weights_only=True)` to prevent arbitrary Python object deserialization via standard `pickle`.

---

## 4. Production Container Deployment Guidelines

```bash
docker run -d \
  --name vista-inspection-prod \
  --restart always \
  --read-only \
  --user 10001:10001 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  -v /opt/vista/checkpoints:/app/checkpoints:ro \
  -v /opt/vista/logs:/var/log/vista:rw \
  -p 8000:8000 \
  vista-vlm:5.0.0
```

### Mandatory Operational Controls
- **Non-Root Execution**: Runs under UID `10001` (`appuser`).
- **Read-Only Root Filesystem**: Write access is forbidden across the container root, preventing malicious script injection.
- **Drop All Capabilities**: Linux capabilities dropped; privilege escalation disabled.
