# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research paper

The paper describing this system is saved locally at `docs/magellan_paper.html` with a structured outline and search index at `docs/magellan_paper_outline.md`. Key sections for implementation reference:
- **§3.2 + Appendix C.2** — MAGELLAN algorithm and hyperparameters (Table 6)
- **Appendix C.1 + Table 5** — SAC hyperparameters and LoRA/adapter setup
- **Appendix A** — Little-Zoo environment mechanics and goal space construction
- **§4 + Appendix D** — Experimental results and baselines to compare against

## What this project does

MAGELLAN (MetAcognitive GEneralization of Learning progress in LANguage model agents) trains LLM-based RL agents on the [LittleZoo](https://github.com/flowersteam/littlezoo) environment using Soft Actor-Critic (SAC). The key innovation is the goal sampler: instead of picking goals randomly, MAGELLAN estimates per-goal Learning Progress (LP) — the absolute change in predicted success rate between a delayed and current SR estimator — and biases sampling toward goals where the agent is improving.

## Setup

Dependencies are managed with `uv` (Python 3.12). The project requires two external packages installed from source:
- **Lamorel** — distributed LLM inference/training framework that wraps HuggingFace models
- **LittleZoo** — the RL environment

```bash
uv sync
```

The LLM model weights must be available locally (e.g. `/LLMs/flan-t5-base`). The path is set per config file via `lamorel_args.llm_args.model_path`.

## Running training

Training is launched via `lamorel_launcher`, which handles distributed setup (1 RL process + N LLM processes):

```bash
# Random goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_random \
  rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/random rl_script_args.seed=0

# MAGELLAN goal sampling
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan \
  rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan rl_script_args.seed=0
```

Other samplers: `local_gpu_config_online` (Online-ALP) and `local_gpu_config_ek_online` (EK-Online-ALP).

Resume from checkpoint by appending `rl_script_args.loading_path=outputs/magellan/10000`.

On HPC clusters, use the SLURM scripts in `configs/little_zoo/` (e.g. `sbatch configs/little_zoo/magellan.sl`).

## Architecture

### Lamorel integration

The agent is a `Caller` object (`lamorel.Caller`). All LLM forward passes and weight updates go through Lamorel's message-passing interface between the RL process and LLM worker processes. Custom behaviors are defined via three extension points:
- **`BaseModuleFunction`** (in `models.py`) — custom forward pass heads attached to the LLM
- **`BaseUpdater`** (in `updater.py`) — custom gradient update logic dispatched on LLM workers
- **`BaseModelInitializer`** (in `initializer.py`) — model setup and LoRA adapter initialization

### LoRA adapter scheme

`PeftInitializer` attaches four named LoRA adapters to the LLM:
- `default` — shared by the policy (actor) and critic
- `critic_target` — frozen copy of critic, updated via polyak averaging
- `sr_adapters` — current SR (success rate) estimator for MAGELLAN
- `delayed_adapters` — lagged copy of SR estimator (frozen, updated by copying past weights into a buffer)

This means the same LLM backbone serves all roles simultaneously by switching adapters.

### Module functions (`models.py`)

- `LogScoringModuleFn` — actor; computes log-probability of each candidate action token sequence
- `ValueHeadModuleFn` — critic; MLP (hidden→1024→1024→1) on top of the last hidden state
- `SRHeadModuleFn` — MAGELLAN SR estimator; MLP (hidden→128→1) with sigmoid output, trained with binary cross-entropy against episode success

### Goal samplers (`goal_sampler.py`)

All samplers share the `GoalSampler` base class. LP is computed as `|sr_current - sr_delayed|` and used as a sampling weight (with ε-greedy exploration):
- `RandomGoalSampler` — uniform random
- `OnlineGoalSampler` — per-goal LP from a sliding success-rate buffer
- `EKOnlineGoalSampler` — same but operates on 4 goal-type buckets (grasp, grow_plants, grow_herbivores, grow_carnivores)
- `MAGELLANGoalSampler` — LP estimated by the LLM-based SR head; delayed weights are maintained via `SACUpdater.weights_buffer` (a deque of past checkpoint snapshots)

### Training loop (`magellan/main.py`)

The main loop alternates between `collect_trajectories` (env rollouts using `agent.custom_module_fns`) and SAC updates dispatched via `agent.update`. Two update functions are dispatched depending on `func` kwarg:
- `sac_update` — critic + policy gradient update with automatic α tuning
- `sr_update` — MAGELLAN SR head update (BCE loss against episode return)

Checkpoints are saved as numbered directories under `output_dir` (e.g. `outputs/magellan/10000/`) containing `model.checkpoint`, optimizer states, `replay_buffer.pkl`, `goal_sampler.pkl`, and (for MAGELLAN) `goal_buffer.pkl`/`success_buffer.pkl`.

### Environment (`environment.py`)

`VectorizedEnv` wraps multiple `LittleZoo` instances. Goals are text prompts (e.g. `"Goal: Grasp apple\nYou see: ...\nAction: "`). `generate_goals` constructs the full goal space and applies a distribution filter — the `goals_distribution` config key controls how many goals per category (impossibles, grasp, grow_plants, grow_herbivores, grow_carnivores) are included in the training set.
