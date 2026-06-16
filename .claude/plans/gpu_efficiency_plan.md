# GPU Efficiency Plan — dev config

## Problem

VRAM usage oscillates between ~21% (collection phase) and ~91% (gradient phase).
The low phase represents wasted GPU capacity; the high phase is near the limit.

## What we can and cannot touch

- `gradient_batch_size` — **leave at 64**. Peak VRAM is already near 17 GB max;
  larger gradient batches increase the activation graph proportionally.
- `minibatch_size` (llm_args) — **leave at 256**. 512 caused OOM during gradient
  passes (activation graph + scoring batch coexist).

## Changes

### 1. `nb_updates: 1` → `2`

**Memory impact:** none. Each update step frees its activation graph before the next
forward pass, so peak VRAM is unchanged. The high-VRAM phase simply lasts longer.

**Speed impact:** 2× gradient steps per collection cycle. SAC is off-policy, so
replay buffer samples can be reused. Better sample efficiency at zero extra memory
cost.

**Observed result:** grasp converged ~2× faster in episode count (step 2004 vs 4009
in the prior run); grow_plants exceeded 5% for the first time (9.375%). Wall-clock
time increased from ~1.5h to ~2.5h at 10k episodes.

### 2. `number_envs: 32` → `64`

**Memory impact:** collection-phase only (low phase, currently 21% VRAM). Larger
inference batches get chunked at `minibatch_size: 256` by Lamorel, so the LLM
scoring batch size is unchanged. CPU/RAM usage increases proportionally.

**Speed impact:** 2× environments stepped per wall-clock second → replay buffer
fills twice as fast → `update_freq: 64` is reached sooner → more training per hour.

**Observed result:** GPU compute utilization became much more stable (fewer and
shorter idle dips). Combined with nb_updates=2, this contributed to the improved
learning signal.

### 3. `buffer_size: 100000` → `250000`

**Memory impact:** replay buffer lives in CPU RAM, not VRAM. No GPU impact.

**Rationale:** at 40k episodes with 64 envs and n_steps=3, the run generates ~7.68M
transitions total. A 100k buffer turns over every ~8 update cycles (too fast —
early experience is lost quickly). 250k holds ~16 update cycles of history,
giving gradient updates a more diverse cross-section of the policy's trajectory.

### 4. `num_episodes: 10000` → `40000`

**Rationale:** 10k episodes was sufficient to confirm the pipeline works and observe
grasp convergence. grow_herbivores only briefly appeared (1.56% peak) before
collapsing to 0%, and grow_carnivores never moved. 40k episodes (~10h) gives the
policy enough gradient steps to climb the harder goal curriculum and produce a
meaningful signal on whether MAGELLAN's LP sampling helps beyond grasp.

## Files changed

- `configs/little_zoo/local_gpu_config_magellan_dev.yaml`
