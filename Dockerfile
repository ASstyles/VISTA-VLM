# Multi-stage production container for VISTA-VLM V7.0
# Hardened, non-root user execution with healthcheck and minimal attack surface

FROM python:3.11-slim as builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY vista_vlm ./vista_vlm
RUN pip install --no-cache-dir --upgrade pip build wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m build --wheel --no-isolation -o /wheels

FROM python:3.11-slim as runtime

# Create dedicated non-root user and group
RUN groupadd -g 10001 vista && \
    useradd -u 10001 -g vista -s /bin/bash -m vista

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Setup application directories with strict owner permissions
RUN mkdir -p /app/checkpoints /app/benchmarks /app/data && \
    chown -R vista:vista /app

USER vista:vista

EXPOSE 8000

# Continuous container liveness verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

# Launch production ASGI server
CMD ["uvicorn", "vista_vlm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

