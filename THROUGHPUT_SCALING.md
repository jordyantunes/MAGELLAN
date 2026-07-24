# Throughput Scaling Findings

> **Update 2026-07-24:** scaling `number_envs` to 128 worked (~8,000–8,500 ep/hr) and
> shifted the bottleneck to the SAC update itself — see `FP32_BOTTLENECK.md` for the
> current time budget and the TF32/mask/empty_cache changes that followed.

How to spend spare GPU capacity (e.g. RTX 4090, 24 GB) productively. Follow-up to
`VRAM_TUNING.md`, which established that scaling batch sizes up made training ~1.9× *slower*.
Verified against the lamorel launcher/source installed in `.venv` (v0.3) and the
[lamorel README](https://github.com/flowersteam/lamorel), 2026-07.

---

## What the three batch knobs actually do

| Knob | Consumed at | Effect |
|---|---|---|
| `lamorel_args.llm_args.minibatch_size` | rollout scoring | Chunk size for no-gradient forward passes. Each env step scores `number_envs × ~10` (prompt, action) pairs; this only sets how many go per chunk. Raising it creates **no extra work** — it just reduces chunk count. |
| `rl_script_args.minibatch_size` | `rb.sample()` in `magellan/main.py` | The real SAC batch: replay transitions per gradient update. Each state expands to all ~10 candidate actions, so 512 states ≈ 5,120 sequences through the LLM **with gradients**. Doubling it doubles GPU work per update. |
| `rl_script_args.gradient_batch_size` | `SACUpdater` (`magellan/updater.py`) | Gradient-accumulation chunk: the SAC batch is processed in `ceil(minibatch_size / gradient_batch_size)` chunks. Pure VRAM knob — does not change the math or the total compute. |

## Why bigger batches = slower

Updates fire on a fixed cadence: every `update_freq: 64` episodes → `nb_updates: 2` gradient
steps. Episode collection is CPU-bound (see GPU-util troughs in `VRAM_TUNING.md`). Doubling the
SAC batch doubles GPU-seconds per update cycle while collecting **zero** extra episodes — the
~1.9× episodes/hr drop (4,040 → 2,170) is just that arithmetic.

**Tuned throughput setting (both 5080 and 4090):**
`lamorel minibatch_size: 256` / `gradient_batch_size: 64` / `rl minibatch_size: 256` → ~10 GB peak, ~4,000 ep/hr.

---

## Option 1 — Scale `number_envs` (one faster run)

`number_envs` is the knob that drives episode collection rate. More envs = more episodes per
wall-clock unit *and* bigger rollout scoring batches (which is where spare VRAM helps).

```bash
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan_4090 \
  rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan_96env rl_script_args.seed=0 \
  rl_script_args.number_envs=96 rl_script_args.update_freq=96 \
  rl_script_args.gradient_batch_size=64 rl_script_args.minibatch_size=256 \
  lamorel_args.llm_args.minibatch_size=256
```

### ⚠️ Hard constraint: `update_freq` must be ≥ `number_envs`

`collect_trajectories` (`magellan/main.py:61`) loops `range(nb_steps // nb_envs)` where
`nb_steps=update_freq` and `nb_envs=number_envs`. If `number_envs > update_freq`, integer division
floors to **0** — the loop body never runs, zero episodes are ever collected, and training spins
forever printing `0/500000 episodes done.` with an instant `0it` tqdm bar. It doesn't crash, so it
looks "stuck" rather than erroring — the only symptom is the episode counter never leaving 0. Found
this the hard way going from `number_envs: 64` (worked, `64 // 64 == 1`) to `number_envs: 96` with
`update_freq` left at its old value of 64 (`64 // 96 == 0`, hung for ~7 min until killed).

**Fix:** bump `update_freq` to match `number_envs` (e.g. both at 96) whenever you raise
`number_envs`. Note this changes the SAC update cadence — updates now fire every 96 episodes
instead of every 64 — so it's not a purely free scaling knob, but no ill effects were observed at
96/96.

Caveats:
- **CPU wall**: env stepping is CPU-side. If GPU-util troughs get deeper/longer at 128, back off
  to 96. Watch with `nvidia-smi -l 1`.
- **Never change `number_envs` on a resumed run** — it is baked into the pickled
  `replay_buffer.pkl` (constructor arg of `NStepReplayBuffer`, used for per-env n-step
  bookkeeping). Fresh runs only.
- Keep `lamorel minibatch_size` at 256–512 and let scoring chunk; 128 envs × ~10 actions ≈ 1,280
  pairs in one chunk would blow past 24 GB at ~20 MB/item.

### Measured result (RTX 4090, `number_envs: 96` / `update_freq: 96`, batch 256/64)

Verified via MLflow `test/grasp` step timestamps (run `2a717e4c`), checked at two points to rule
out a short-window fluke:

| Checkpoint | Episodes | Elapsed | ep/hr |
|---|---|---|---|
| ~1h in | 0 → 6,007 | 1.03h | 5,814 |
| ~1.9h in | 0 → 11,000 | 1.88h | 5,858 |

Stable at **~5,850 ep/hr** — a **+38%** gain over the best 64-env 4090 run (4,231 ep/hr) and +45%
over the 5080 baseline (4,039 ep/hr). `nvidia-smi` trace over a 39-min window during this run
(`gpu_report_num_envs_test.csv`) showed 84.1% average utilization, only 2.3% of samples at 0%
(brief single-second stalls around checkpoint saves/tests), vs. the frequent deep troughs seen at
64 envs in `VRAM_TUNING.md`.

## Option 2 — Two seeds concurrently (double experiment throughput)

Two launches from the same config. Three things must differ: `seed`, `output_dir`, and the
rendezvous port. Both runs need the lean tuned batch settings (2 × ~10 GB fits in 24 GB).

```bash
# Terminal 1
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan_4090 \
  rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan rl_script_args.seed=0 \
  rl_script_args.gradient_batch_size=64 rl_script_args.minibatch_size=256 lamorel_args.llm_args.minibatch_size=256

# Terminal 2 — different seed, output_dir, and port
python -m lamorel_launcher.launch --config-path configs/little_zoo/ --config-name local_gpu_config_magellan_4090 \
  rl_script_args.path=magellan/main.py rl_script_args.output_dir=outputs/magellan_s1 rl_script_args.seed=1 \
  rl_script_args.gradient_batch_size=64 rl_script_args.minibatch_size=256 lamorel_args.llm_args.minibatch_size=256 \
  lamorel_args.accelerate_args.main_process_port=29502
```

- **The port override is mandatory** — without it the second run tries to bind 29501 (held by the
  first) and hangs/crashes at startup.
- MLflow needs nothing: both land in the same experiment (named after the goal sampler) as runs
  `seed0` / `seed1`; each logs its timestamped `output_dir` as a param.
- CPU/RAM roughly double: each run steps 64 train envs **plus** creates 256 test + 256 eval envs.
  Expect each run somewhat slower than ~3,900 ep/hr solo; combined > solo is still a win.
- Don't combine with Option 1 (128 envs × 2 runs slams both VRAM and CPU).

### Two GPUs in one box (5080 + 4090)

Lamorel maps single-machine processes to `cuda:{RANK-1}` (`lamorel/init_distributed_setup.py`),
so both the RL process and the LLM server land on device 0 — two runs would pile onto the same
card. Pin each run instead:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lamorel_launcher.launch ... rl_script_args.seed=0 ...
CUDA_VISIBLE_DEVICES=1 python -m lamorel_launcher.launch ... rl_script_args.seed=1 \
  lamorel_args.accelerate_args.main_process_port=29502 ...
```

(`nvidia-smi -L` to see which index is which card.) Each run then gets a whole GPU and needs no
batch shrinking — expect close to ~4,000 ep/hr each.

---

## Lamorel internals verified along the way

- `lamorel_launcher/launch.py` passes everything under `lamorel_args.accelerate_args` **verbatim
  to `accelerate launch`'s arg parser** — any `accelerate launch` flag is a valid config key
  there. `main_process_port` (undocumented in the lamorel README) becomes `MASTER_PORT` for the
  torch.distributed rendezvous; it also overrides the port in `default_config.yaml`.
- Two experiments on different ports form fully separate process groups on 127.0.0.1 — no
  cross-talk.
- Multiple LLM servers on one machine is an intended lamorel pattern (README:
  `model_parallelism_size` limits GPUs per server; our configs already set it to 1).
- README confirms `llm_args.minibatch_size` is "batch size per forward passes, adapt this number
  to your GPU memory" — a VRAM chunking knob, not a work-creation knob.
- Running two *separate* lamorel experiments concurrently is not documented anywhere — it works
  by architecture (separate process groups), not by explicit support.

## Decision guide

| Goal | Do this |
|---|---|
| One result sooner | Option 1: `number_envs` 96–128 **with `update_freq` bumped to match**, tuned batch knobs — confirmed ~5,850 ep/hr at 96/96 |
| Multiple seeds (you'll need them anyway) | Option 2: two concurrent runs, distinct port/seed/output_dir |
| Both cards in one box | Option 2 with `CUDA_VISIBLE_DEVICES` pinning, one run per GPU |
| Spend VRAM on capacity instead | flan-t5-large fits 24 GB in full precision (see `COMPATIBILITY.md` for bitsandbytes caveats) |
