# VRAM Tuning Findings

> **Update 2026-07-24:** at `number_envs: 128` the bottleneck has moved from env
> collection to the SAC update's GPU math (fp32, tensor cores unused) — see
> `FP32_BOTTLENECK.md` before acting on the batch-size guidance below.

Hardware used for initial tuning: RTX 5080 (16 GB GDDR7, sm_120 / Blackwell)  
Model: `google/flan-t5-base` (~250 MB weights), full precision (fp32/bf16), no quantization  
LoRA: 4 adapters (default, critic_target, sr_adapters, delayed_adapters), r=16 α=32

---

## What drives VRAM usage

Two separate forward-pass codepaths with different batching controls:

| Phase | Config key | Controls |
|-------|-----------|---------|
| **Inference** (env rollout / scoring) | `lamorel_args.llm_args.minibatch_size` | Max (context, action) pairs per forward pass, no gradients |
| **Training** (SAC critic/policy update) | `rl_script_args.gradient_batch_size` | Replay states sampled per update; the updater computes `_batch_size = sum(len(possible_actions))` and bypasses `minibatch_size` entirely |

The effective gradient batch fed to the LLM is approximately:

```
gradient_items ≈ gradient_batch_size × avg_actions_per_state   (~10 for LittleZoo)
```

---

## VRAM observations (flan-t5-base, RTX 5080 16 GB)

| `minibatch_size` | `gradient_batch_size` | Peak VRAM | Result |
|---|---|---|---|
| 1024 | 128 | ~13 GB (inference alone) | OOM in LoRA forward |
| 64 | 16 | ~2.7 GB | Stable, ~17% utilization — too slow |
| 256 | 64 | ~10 GB peak | Stable, good utilization |
| 256 | 128 | ~15 GB peak | OOM during SAC update |
| 256 | 96 | ~11–12 GB peak | Stable — tuned setting |

---

## Scaling rules for other machines

**Baseline memory** (model weights + 4 LoRA adapters + optimizer states): ~2–3 GB for flan-t5-base fp32.

**Inference peak** scales roughly linearly with `minibatch_size`:
- 256 → ~5–6 GB during rollout
- Rule of thumb: ~20 MB per item in the batch

**Training peak** scales with `gradient_batch_size × avg_actions (~10)`:
- 64 × 10 = 640 gradient items → ~10 GB peak
- 96 × 10 = 960 gradient items → ~12 GB peak
- 128 × 10 = 1280 gradient items → OOM at 15 GB on 16 GB card

**Recommended starting points by VRAM:**

| VRAM | `minibatch_size` | `gradient_batch_size` | Expected peak |
|------|---|---|---|
| 8 GB | 128 | 32 | ~6–7 GB |
| 16 GB | 256 | 96 | ~11–12 GB |
| 24 GB (3090/4090) | 512 | 128 | ~21–22 GB peak, thin margin |
| 40 GB (A100) | 1024 | 384 | ~32–36 GB |

These are starting points — tune up if stable, tune down if OOM.

---

## RTX 4090 (24 GB) actual findings

Initial guess of `minibatch_size: 512` / `gradient_batch_size: 192` (doubling both knobs from the
16 GB tuned setting) **OOM'd immediately** — extrapolating linearly from the 16 GB data
(96 → ~12 GB, 128 → OOM at 15 GB), 192 alone implies ~24 GB for the training peak, before baseline
or inference are added. The doc's original 24 GB row was too optimistic for a straight 2× jump on
both knobs at once.

Dialed back to `gradient_batch_size: 128` (keeping `minibatch_size: 512`) and added
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. Captured via
`nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1`
during a live run:

- VRAM oscillates every 1–3 s between ~2.7–4 GB (rollout/inference troughs) and ~18–22 GB (SAC
  update peaks) — the two-codepath alternation described above, on a cadence consistent with
  `update_freq: 64`.
- Peak samples showed as little as **~2.3–2.8 GB free out of 24 GB total** — the run survived the
  observed window but the margin is thin. `avg_actions_per_state` is an average, not a cap, so a
  batch with more actions than usual could push past 24 GB.
- GPU compute utilization was generally 70–100%, with dips to 20–50% (and a few sub-20% blips)
  that track the low-VRAM troughs — likely CPU-bound env stepping across the 64 parallel envs
  between LLM calls, not GPU starvation.

**Conclusion:** `minibatch_size: 512` / `gradient_batch_size: 128` is a viable but tight setting on
24 GB. For more safety margin before pushing higher again, consider dialing `gradient_batch_size`
down to ~112 rather than trying 160+ without more headroom evidence.

---

## VRAM headroom does not mean more throughput

Fitting a bigger batch in VRAM is not the same as training faster. Compared three MLflow runs by
episodes/hour (via `test/grasp` step timestamps):

| Run | GPU | `minibatch_size` / `gradient_batch_size` | Episodes/hr |
|---|---|---|---|
| `2b60142f` | RTX 5080 | 256 / 64 | ~4,040 |
| `d776f056` | RTX 4090 | 512 / 128 | ~2,170 |
| `84736cd4` | RTX 4090 | 256 / 64 | ~3,900 |
| `2a717e4c` | RTX 4090 | 256 / 64, `number_envs: 96` | ~5,850 |

Doubling `minibatch_size`/`gradient_batch_size` on the 4090 (to use the extra VRAM headroom) made
training ~1.9× **slower** in episodes/hr, not faster. Reverting to the 5080-tuned 256/64 setting
on the same 4090 recovered throughput to roughly match the 5080 (~3,900 vs ~4,040 ep/hr) —
i.e. the 4090 isn't meaningfully faster than the 5080 for this workload at matched batch size
either.

**Why:** `number_envs` (rollout parallelism) wasn't scaled up alongside the batch sizes. Episode
throughput here is bottlenecked by env rollout / episode collection (see the CPU-bound dips noted
above), not by how large a gradient batch the GPU can chew through in one SAC update. Growing
`gradient_batch_size` just makes each update cycle heavier without increasing how many episodes
get collected between updates — pure overhead.

**Takeaway:** don't scale `minibatch_size`/`gradient_batch_size` up just because VRAM allows it.
If you want to spend extra VRAM/compute headroom productively, try scaling `number_envs` instead —
that's the knob that actually drives episode collection rate. Confirmed: `number_envs: 96` (with
`update_freq` bumped to 96 to match — see the hard constraint in `THROUGHPUT_SCALING.md`) reached
~5,850 ep/hr, +38% over the best 64-env run. See `THROUGHPUT_SCALING.md` for concrete recipes (more
envs, two concurrent seeds, GPU pinning) and the lamorel details behind them.

---

## Larger models

Larger models (flan-t5-large, flan-t5-xl) will push baseline up significantly. Before using them:
- Re-enable `load_in_4bit: true` in the config
- Resolve the accelerate/bitsandbytes conflict (see COMPATIBILITY.md)
- Reduce both batch sizes accordingly and tune up from there

---

## Other memory culprits (system RAM, not VRAM)

- **`generate_goals` RAM**: Before the reservoir-sampling fix (see COMPATIBILITY.md), the full
  O(n⁵) goal space (~19M strings) was materialized before subsampling — 18.4 GB peak. After
  fix: ~46 MB. If you see the process OOM before any CUDA work starts, this is the cause.
- **Replay buffer**: `buffer_size: 500000` grows over time. Monitor with `psutil` if system
  RAM is tight on the machine.

---

## Config keys to change per machine

In `configs/little_zoo/<your_config>.yaml`:

```yaml
lamorel_args:
  llm_args:
    minibatch_size: 256   # scale with VRAM

rl_script_args:
  gradient_batch_size: 96  # scale with VRAM; keep gradient_items < VRAM*80/1000
  minibatch_size: 256      # SAC replay sample size; keep <= lamorel minibatch_size
```
