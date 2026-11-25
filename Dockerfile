FROM ghcr.io/astral-sh/uv:0.7.13 AS uv
FROM python:3.11-slim AS base
COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/services/inference:/workspace/services/edge-agent:/workspace/ai:/workspace/packages/core
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/workspace/.venv/bin:$PATH"
COPY ai ai
COPY packages packages
COPY services services
RUN useradd --uid 10001 --create-home platform && install -d -m 0700 -o platform /var/lib/viweb-demo /var/lib/viweb-training
USER platform

CMD ["uvicorn", "vi_inference.main:app", "--host", "0.0.0.0", "--port", "8001"]

FROM base AS yolo
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN uv pip install --python /workspace/.venv/bin/python --torch-backend cpu ultralytics==8.4.216 "pyyaml>=6,<7"
ADD --checksum=sha256:0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 https://huggingface.co/Ultralytics/YOLO11/resolve/5599de6/yolo11n.pt /opt/viweb/yolo11n.pt
RUN chmod 0644 /opt/viweb/yolo11n.pt
ENV VI_TRAINING_WEIGHTS=/opt/viweb/yolo11n.pt
USER platform

FROM base AS default
