# Apple Silicon / MLX Compatibility Plan

## Executive Summary

**Not feasible without major architectural changes.** Risk: **HIGH**. Unlike AMD/ROCm — which transparently remaps the `cuda` namespace — Apple Silicon offers two distinct paths: **MPS** (PyTorch's Metal backend) and **MLX** (Apple's own array framework). Neither is a drop-in replacement for CUDA, and both hit the same fundamental wall: `torch.distributed` + DDP, the backbone of Lamorel's multi-process design, does not run on MPS or MLX. This is not a patching problem; it is an architectural mismatch.

---

## Clarification: MPS vs. MLX

| | **MPS** | **MLX** |
|---|---|---|
| What it is | PyTorch's Metal Performance Shaders backend (`device="mps"`) | Apple's own ML framework (like JAX/NumPy) |
| PyTorch compatible | Yes — same API, different device | No — completely different API |
| HF Transformers | Supported for inference | Via `mlx-lm`, inference only |
| `torch.distributed` | Not supported | Not applicable |
| Relevant path | Would require patching existing code | Would require rewriting the entire stack |

The analysis below focuses on **MPS** as the only plausible path (MLX would require a complete rewrite of Lamorel, the updater, the training loop, and the module functions).

---

## Blockers

| File | Line | Issue | Severity |
|---|---|---|---|
| `lamorel/init_distributed_setup.py` | 11 | `ACCELERATE_TORCH_DEVICE = f"cuda:{...}"` — hardcoded CUDA device string, not remapped on MPS | CRITICAL |
| `lamorel/server/server.py` | 88 | `torch.cuda.device_count()` — returns 0 on MPS; no MPS equivalent | CRITICAL |
| `lamorel/server/llms/hf_llm.py` | 54 | `torch.cuda.mem_get_info(f'cuda:{_device}')` — crashes on MPS, no namespace remapping | CRITICAL |
| `lamorel/server/server.py` | 61 | `DDP(self._model, process_group=self._llm_group, ...)` — DDP over `gloo` requires CPU tensors; MPS tensors cannot participate in gloo collectives | CRITICAL |
| `lamorel/server/llms/hf_llm.py` | 401–407 | `torch.cuda.synchronize()` and `torch.cuda.empty_cache()` — no-op on MPS would be acceptable, but MPS offers no equivalent (`torch.mps.synchronize()`, `torch.mps.empty_cache()` exist but are not called) | HIGH |
| `magellan/updater.py` | 244, 326 | `torch.cuda.empty_cache()` — MPS has `torch.mps.empty_cache()`, not remapped | HIGH |
| `pyproject.toml` | 8 | `bitsandbytes>=0.49.2` — no Apple Silicon support whatsoever | HIGH |

---

## The Core Architectural Problem

Unlike ROCm, which re-uses PyTorch's CUDA namespace transparently, **MPS is a separate device** — `device("mps")` not `device("cuda")`. This means:

1. **`torch.cuda.*` calls do not work at all.** Every `torch.cuda.mem_get_info`, `torch.cuda.device_count`, `torch.cuda.synchronize`, `torch.cuda.empty_cache` call raises an error or returns wrong results. There is no compat shim.

2. **`torch.distributed` does not support MPS as a backend.** Lamorel spawns 1 RL process + N LLM processes and coordinates them using `gloo` distributed collectives (`gather_object`, `broadcast_object_list`). On CUDA/ROCm, `gloo` moves tensors to CPU for collectives and then back. On MPS, this pathway is not implemented — `torch.distributed` will refuse to initialize on MPS devices.

3. **`DDP` does not support MPS.** `DistributedDataParallel` requires a backend (`gloo`, `nccl`, or `mpi`) and none of them support MPS tensors. Running DDP with `use_cpu=True` would work (falls back to CPU) but defeats the purpose of using Apple Silicon.

4. **Accelerate's MPS support is inference-only.** `Accelerator()` will detect MPS and set `device="mps"`, but distributed training (`n_processes > 1`) is not supported on MPS. The multi-process Lamorel launch would fail at `init_distributed_setup`.

---

## What Does Work on MPS (if single-process)

If Lamorel were replaced with a single-process inference/update loop:

- All `.to(self.device)` calls — already device-agnostic, would work with `device="mps"`
- `torch.nn.Linear`, `torch.nn.Sequential` — fully supported on MPS
- HuggingFace forward passes — supported on MPS
- PEFT LoRA adapters — work on MPS for inference; gradient training is partially supported
- `torch.optim.Adam` — supported on MPS

---

## Unknowns / Needs Testing

- Whether PEFT's multi-adapter `set_adapter()` + `add_adapter()` scheme works under MPS gradients (no known issues, but untested)
- Whether `gradient_checkpointing_enable()` is stable on MPS
- `bfloat16` support on MPS — M2 and later support it; M1 does not

---

## Dependency Analysis

| Package | Status on Apple Silicon |
|---|---|
| **PyTorch** | MPS backend available via standard `pip install torch`; no special wheel needed |
| **bitsandbytes** | Complete blocker — no Apple Silicon support, no workaround |
| **accelerate** | MPS supported for single-process inference; multi-process training not supported |
| **peft** | Works on MPS for inference; training compatibility partial |
| **transformers** | MPS supported |
| **torch.distributed** | Not supported on MPS |

---

## Implementation Plan

These are ordered by impact. **Item 1 is not patchable** — it requires redesigning Lamorel's distributed architecture for the single-machine case.

### 1. Replace Lamorel's multi-process architecture with single-process (CRITICAL — blocks everything)

The `DDP` + `torch.distributed` design in Lamorel fundamentally requires CUDA or CPU collectives. For MPS to work, Lamorel would need to run in a single process (no RL/LLM process split), with the LLM called inline. This is a significant fork of Lamorel, not a patch.

### 2. Replace all `torch.cuda.*` calls (CRITICAL — runtime crashes)

Replace or guard every CUDA-specific call:

| Old | New |
|---|---|
| `torch.cuda.mem_get_info(f'cuda:{_device}')` | Use `psutil.virtual_memory().available` or skip for MPS |
| `torch.cuda.device_count()` | `1` on MPS (single unified device) |
| `torch.cuda.empty_cache()` | `torch.mps.empty_cache()` |
| `torch.cuda.synchronize()` | `torch.mps.synchronize()` |
| `ACCELERATE_TORCH_DEVICE = f"cuda:{...}"` | `ACCELERATE_TORCH_DEVICE = "mps"` |

### 3. Make `bitsandbytes` optional (HIGH — same fix as AMD plan)

Same as AMD plan item 1: move to an optional `[quantization]` extra, guard the import in `initializer.py` behind `if use_4bit`.

### 4. Set device strings to `"mps"` throughout configs

All `"cuda:N"` device strings in YAML configs would need to become `"mps"` (MPS has a single unified device, no ordinals).

### 5. Test path

Only viable if item 1 is resolved. Use the dev config (`load_in_4bit: false`, `use_gpu: true`) in a single-process Lamorel mode as the first target.

---

## Recommendation

**Do not pursue Apple Silicon / MPS support** unless Lamorel's multi-process distributed architecture is first replaced. The AMD path is viable with 2 targeted patches; the MPS path requires redesigning the RL↔LLM process communication layer. For local development on a Mac, the CPU path (`use_gpu: false`) is the practical option today.

---

## Audit Notes

- Audited: `magellan/models.py`, `magellan/updater.py`, `magellan/initializer.py`, `magellan/main.py`, `magellan/goal_sampler.py`, all YAML configs, and all Lamorel source files
- Reference: `docs/amd_gpu_compatibility.md` for comparison
- MPS documentation: `torch.backends.mps.is_available()`, `torch.mps.*` namespace
