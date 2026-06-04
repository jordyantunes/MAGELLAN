# CUDA 12.8 devel gives us nvcc + cuDNN for bitsandbytes; torch 2.12 pulls
# CUDA 13 runtime wheels from PyPI on top of this driver-compatible base.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # HuggingFace cache inside the container (override at runtime with -v)
    HF_HOME=/app/.cache/huggingface

# System deps: git (for uv git sources + SSH), openssh-client for private forks
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Route GitHub HTTPS through SSH so uv can fetch the private forks.
# The SSH key is injected at build time via --secret id=ssh_key.
RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN --mount=type=secret,id=ssh_key,dst=/root/.ssh/id_ed25519 \
    git config --global url."git@github.com:".insteadOf "https://github.com/"

# Copy lockfile and project metadata first for layer caching
COPY pyproject.toml uv.lock ./

# Install Python 3.12 and all dependencies (frozen = exact lockfile versions)
# --no-install-project skips installing the magellan package itself for now
RUN --mount=type=secret,id=ssh_key,dst=/root/.ssh/id_ed25519 \
    uv sync --frozen --no-install-project

# Copy the rest of the project
COPY . .

# Install the magellan package itself
RUN uv sync --frozen

# Activate the venv for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Outputs and HF cache are external; declare them as mount points
VOLUME ["/app/outputs", "/app/.cache/huggingface"]

# Default: random-sampler training run. Override CMD or pass args directly.
CMD ["python", "-m", "lamorel_launcher.launch", \
     "--config-path", "/app/configs/little_zoo/", \
     "--config-name", "local_gpu_config_random", \
     "rl_script_args.path=/app/magellan/main.py", \
     "rl_script_args.output_dir=/app/outputs/random", \
     "rl_script_args.seed=0"]
