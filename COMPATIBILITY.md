# Compatibility Notes

This file documents dependency version pins and code patches applied to make MAGELLAN
run on modern hardware/software, along with the reason each change was necessary.

---

## Dependency pins

### `gymnasium==0.29.1`

**Problem:** gymnasium 1.0 changed the `reset()` API so that all wrapper layers
explicitly pass `seed` and `options` kwargs down the wrapper chain. The underlying
`PlayGroundNavigationV1.reset()` (from the Playground environment used by LittleZoo)
was written for the pre-1.0 API and does not accept those kwargs, raising:

```
TypeError: PlayGroundNavigationV1.reset() got an unexpected keyword argument 'seed'
```

**Fix:** Pin to `gymnasium==0.29.1`, the last release before the breaking API change.
LittleZoo has no gymnasium version constraint of its own, so this pin is safe.

---

### `transformers==4.44.2`

**Problem:** transformers 5.x introduced a `_finalize_model_loading` method that calls
`tie_weights(missing_keys=..., recompute_mapping=False)`. That new `tie_weights`
implementation uses `torch.equal` to check whether weights are already tied. When
lamorel calls `from_pretrained` inside an `accelerate.init_empty_weights()` context
(to run `infer_auto_device_map` without allocating real memory), all tensors are Meta
tensors and `torch.equal` raises:

```
NotImplementedError: aten::equal: attempted to run this operator with Meta tensors
```

**Fix:** Pin to `transformers==4.44.2`, a 4.x release where `tie_weights` does not
use `torch.equal` and is safe to call inside `init_empty_weights()`.

---

## Config changes (`configs/little_zoo/local_gpu_config_random.yaml`)

| Key | Old value | New value | Reason |
|-----|-----------|-----------|--------|
| `lamorel_args.llm_args.seed` | *(absent)* | `null` | lamorel's `HF_LLM.__init__` accesses `args.seed`; omegaconf raises `ConfigAttributeError` if the key is missing from the struct. |
| `lamorel_args.llm_args.model_path` | `/LLMs/flan-t5-base` | `google/flan-t5-base` | The local path does not exist on this machine. Using the HuggingFace Hub identifier lets transformers download weights automatically. |
| `lamorel_args.llm_args.load_in_4bit` | `true` | `false` | With `transformers==4.44.2` + current accelerate, loading a 4-bit model with an explicit `device_map` causes `dispatch_model` to call `model.to(device)`, which bitsandbytes forbids. flan-t5-base (~250 MB weights) fits in full precision on the RTX 5080's 16 GB VRAM, so quantization is not needed. Re-enable when switching to a larger model and after resolving the accelerate/bitsandbytes version conflict. |
| `lamorel_args.llm_args.minibatch_size` | `1024` | `256` | With 1024 (context, action) pairs batched through flan-t5-base in a single forward pass, activation memory alone consumes ~13 GB, causing CUDA OOM. Tuned down to 256 for stable VRAM usage (~3.5 GB baseline, ~10 GB peak on RTX 5080 16 GB). |
| `rl_script_args.gradient_batch_size` | `128` | `64` | During SAC updates the updater computes `_batch_size = sum(len(possible_actions))` across the gradient batch and passes it directly as the LLM `minibatch_size` for gradient-enabled forward passes. 128 states × ~10 possible actions = ~1280 items through T5 with gradients, hitting 15 GB and OOMing. 64 keeps peak VRAM around 10 GB on the RTX 5080 16 GB. Gradient accumulation keeps the effective replay batch size at 256. |

---

## Code patches

### `magellan/environment.py` — reservoir sampling in `generate_goals`

**Problem:** `generate_goals` enumerated the full O(n⁵) goal space (~19M strings for
the default object set) into Python lists before subsampling to the requested
distribution. Peak memory usage was ~18 GB, exhausting system RAM before training
could start.

**Fix:** Replaced the enumerate-then-sample pattern with reservoir sampling
(Algorithm R). Each category maintains a fixed-size reservoir (`distribution[i]`
entries) updated in-place as goals are streamed from the nested loops. Peak memory
dropped from ~18 GB to ~46 MB.

---

### `magellan/utils/generate_prompt.py` — accept extra positional arguments

**Problem:** `tests.py` calls `generate_prompt(obs, goal, 'LittleZoo')` but the
function signature was `generate_prompt(o, g)`, raising `TypeError`.

**Fix:** Changed signature to `generate_prompt(o, g, *args, **kwargs)` to absorb
the unused env-type argument.

---

