# VRAM Tuning Findings

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
| 24 GB (3090/4090) | 512 | 192 | ~16–18 GB |
| 40 GB (A100) | 1024 | 384 | ~32–36 GB |

These are starting points — tune up if stable, tune down if OOM.

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
