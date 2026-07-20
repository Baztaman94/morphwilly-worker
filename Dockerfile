# RunPod Serverless worker image: FaceFusion (CUDA) + the MorphWilly handler.
#
# Build & push (see docs/runpod-serverless.md):
#   docker build -t <registry>/morphwilly-worker:latest worker/
#   docker push  <registry>/morphwilly-worker:latest
#
# Then create a RunPod Serverless endpoint from that image.
#
# NB: FaceFusion install flags and the CUDA base evolve. If the build breaks,
# align the base image's CUDA version with FaceFusion's onnxruntime-gpu wheel.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FACEFUSION_DIR=/app/facefusion

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg curl ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin a known FaceFusion release for reproducible builds.
ARG FACEFUSION_VERSION=3.1.1
RUN git clone --depth 1 --branch ${FACEFUSION_VERSION} \
    https://github.com/facefusion/facefusion.git ${FACEFUSION_DIR}

WORKDIR ${FACEFUSION_DIR}
RUN python -m pip install --upgrade pip \
    && python install.py --onnxruntime cuda --skip-conda

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY handler.py .

# RunPod serverless entrypoint.
CMD ["python", "-u", "handler.py"]
