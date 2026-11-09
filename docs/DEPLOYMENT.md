# VISTA-VLM v4.0 Production Deployment & Operations Guide

## 1. Containerization

VISTA-VLM is packaged as a high-performance, containerized microservice.

### 1.1 Dockerfile
```dockerfile
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies for OpenCV and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and checkpoints
COPY vista_vlm/ /app/vista_vlm/
COPY checkpoints/ /app/checkpoints/
COPY benchmarks/results/ /app/benchmarks/results/

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck using liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["python", "-m", "uvicorn", "vista_vlm.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## 2. Kubernetes Deployment Specification

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vista-vlm-api
  labels:
    app: vista-vlm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vista-vlm
  template:
    metadata:
      labels:
        app: vista-vlm
    spec:
      containers:
      - name: api
        image: vista-vlm:4.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: true
          runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: vista-vlm-service
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: vista-vlm
```

---

## 3. High-Availability & Edge Performance Tuning

1. **Model Warmup**: Foundation models (DINOv2 and CLIP) perform initial graph allocations during the first inference cycle. The `/health/ready` probe verifies warmup completion before routing live traffic.
2. **Text Embedding Caching**: Prompt tokenization is pre-computed in-memory, ensuring verifier operations require $<2\text{ms}$ on CPU.
3. **Hardware Acceleration**: For GPU deployments, set `device = "cuda"` in configuration. This boosts throughput to $>45\text{ FPS}$ on an NVIDIA RTX 4090 or A100.
