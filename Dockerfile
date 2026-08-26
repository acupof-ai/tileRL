# CUDA 13 devel base. Why devel: TileLang JITs CUDA at runtime and needs
# nvcc + CUDA headers; the runtime image lacks them. Why 13:
#   - torch 2.13.0 (pinned in uv.lock) is the PyPI CUDA build, which bundles
#     CUDA 13 userspace libs via pip nvidia-* packages on linux.
#   - tilelang 0.1.13's nvcc extra pins nvidia-cuda-nvcc>=13.0.48, i.e. the
#     wheel ecosystem targets CUDA 13.
# The NVIDIA driver (libcuda.so) is NOT in this image — it comes from the host
# via the NVIDIA container toolkit at runtime.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TILERL_TARGET=cuda

# Ubuntu 22.04 ships python 3.10; the project requires >=3.11. Deadsnakes is
# the standard source. uv goes in via the official installer (no pip yet).
# ponytail: uv version unpinned — a future uv release could change sync
# behavior; upgrade path: pin UV_VERSION in the installer URL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer first: tilelang[fp4] is a main dependency (not a tilerl
# extra), so plain `uv sync` is the equivalent of the requested `--extra gpu`.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Default starts the tiny model (no checkpoint needed). Production overrides
# the command: serve --model qwen38-27b with TILERL_QWEN38_SOURCE set.
CMD ["/app/.venv/bin/tilerl", "serve", "--model", "tiny", "--host", "0.0.0.0"]
