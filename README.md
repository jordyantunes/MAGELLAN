# 🧭 MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces

**MAGELLAN (MetAcognitive GEneralization of Learning progress in LANguage model agents)** is a metacognitive framework designed for Large Language Model (LLM) agents. It enables LLM agents to predict their competence and Learning Progress (LP) online, leveraging semantic relationships between goals to prioritize learning efficiently. By integrating MAGELLAN with online Reinforcement Learning (RL), agents can navigate vast goal spaces adaptively, ensuring efficient learning in high-dimensional and evolving goal spaces.

<div align=center>
  
![magellan (1)](https://github.com/user-attachments/assets/520b3117-829a-44b8-8b9b-72131187c177)

</div>

---

## 🛠 Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). MAGELLAN uses private forks of [Lamorel](https://github.com/jordyantunes/lamorel) and [LittleZoo](https://github.com/jordyantunes/littlezoo) — the lamorel fork includes a fix for a bitsandbytes/accelerate incompatibility (see [COMPATIBILITY.md](COMPATIBILITY.md)).

Since the forks are private, uv needs SSH access to GitHub to fetch them. Make sure your SSH key is registered with GitHub, then:

```bash
# Route GitHub HTTPS URLs through SSH (needed for private repos)
git config --global url."git@github.com:".insteadOf "https://github.com/"

# Install all dependencies (Python 3.12 managed by uv)
uv sync
```

---

## 🚀 Usage

### ⚙️ Configuration

MAGELLAN uses **Hydra** for configuration management. Example configurations can be found in the `configs/` directory.

### 🎯 Training

To train a model using different goal sampling strategies, run one of the following commands:

```bash
# Random goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_random rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/random rl_script_args.seed=0

# Online-ALP goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_online rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/online rl_script_args.seed=0

# EK-Online-ALP goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_ek_online rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/ek_online rl_script_args.seed=0

# MAGELLAN goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan rl_script_args.seed=0
```

### 🔄 Resume Training

To resume training from a checkpoint:

```bash
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan rl_script_args.seed=0 rl_script_args.loading_path=outputs/magellan/10000
```

### 🐳 Docker

#### Build

The forks are private, so the build needs your SSH key to fetch them:

```bash
docker build --secret id=ssh_key,src=$HOME/.ssh/id_ed25519 -t magellan .
```

#### Run locally

```bash
# Random sampler (default)
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v hf_cache:/app/.cache/huggingface \
  magellan

# MAGELLAN sampler
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v hf_cache:/app/.cache/huggingface \
  magellan \
  python -m lamorel_launcher.launch \
    --config-path /app/configs/little_zoo/ \
    --config-name local_gpu_config_magellan \
    rl_script_args.path=/app/magellan/main.py \
    rl_script_args.output_dir=/app/outputs/magellan \
    rl_script_args.seed=0

# Resume from checkpoint
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v hf_cache:/app/.cache/huggingface \
  magellan \
  python -m lamorel_launcher.launch \
    --config-path /app/configs/little_zoo/ \
    --config-name local_gpu_config_magellan \
    rl_script_args.path=/app/magellan/main.py \
    rl_script_args.output_dir=/app/outputs/magellan \
    rl_script_args.seed=0 \
    rl_script_args.loading_path=/app/outputs/magellan/10000
```

Or with Docker Compose:

```bash
docker compose up --build
```

#### Deploy on AWS (EC2 Spot)

EC2 Spot instances give ~70% cost savings. The built-in checkpoint/resume support makes MAGELLAN Spot-safe: if the instance is interrupted, resume from the latest checkpoint.

**Recommended instance types:**

| Instance | GPU | VRAM | Notes |
|---|---|---|---|
| `g5.xlarge` | A10G | 24 GB | Best price/performance starting point |
| `g5.2xlarge` | A10G | 24 GB | More CPU/RAM if the RL process is the bottleneck |
| `p3.2xlarge` | V100 | 16 GB | Cheaper spot market, good alternative |

**Setup steps:**

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name magellan
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag magellan:latest <account>.dkr.ecr.<region>.amazonaws.com/magellan:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/magellan:latest

# 2. On the EC2 instance (after installing NVIDIA Container Toolkit):
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker pull <account>.dkr.ecr.<region>.amazonaws.com/magellan:latest
docker run --gpus all \
  -v /mnt/efs/outputs:/app/outputs \
  -v /mnt/efs/hf_cache:/app/.cache/huggingface \
  <account>.dkr.ecr.<region>.amazonaws.com/magellan:latest
```

Mount an EFS volume at `/mnt/efs` so outputs and model weights persist across Spot interruptions and instance restarts.

---

### 🖥️ HPC Cluster Usage

SLURM job scripts are available for training on HPC clusters:

```bash
# Submit a job with random goal sampling
sbatch configs/little_zoo/random.sl

# Submit a job with MAGELLAN goal sampling
sbatch configs/little_zoo/magellan.sl
```

---

## 📁 Project Structure

- **`magellan/`** – Main source code
  - `main.py` – Entry point for training
  - `environment.py` – Environment-related code
  - `goal_sampler.py` – Goal sampling strategies
  - `models.py` – LLM actor, critic, and LP estimator implementations
  - `updater.py` – SAC and MAGELLAN update logic
  - `initializer.py` – Model initialization utilities
  - `utils/` – Helper functions and utilities
- **`configs/`** – Configuration files for experiments
  - `little_zoo/` – Configurations for the LittleZoo environment

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{gaven2025magellan,
  title={MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces},
  author={Gaven, Loris and Carta, Thomas and Romac, Cl{\'e}ment and Colas, C{\'e}dric and Lamprier, Sylvain and Sigaud, Olivier and Oudeyer, Pierre-Yves},
  journal={arXiv preprint arXiv:2502.07709},
  year={2025}
}
```

---

## 🤝 Contribute

Contributions are welcome! Feel free to open an issue or submit a pull request on [GitHub](https://github.com/LorisGaven/MAGELLAN). 🚀

