# AMD GPU Compatibility Plan

## Executive Summary

Feasible with moderate effort. Risk: **medium**. PyTorch's ROCm build remaps the `cuda` namespace to HIP, so nearly all `torch.cuda.*` calls work transparently. Two real problems exist: `bitsandbytes` and one unconditional `torch.cuda.mem_get_info` call in Lamorel.

---

## Blockers

| File | Line | Issue | Severity |
|---|---|---|---|
| `pyproject.toml` | 8 | `bitsandbytes>=0.49.2` is a hard dependency — its CUDA kernels don't run on AMD | HIGH |
| `lamorel/server/llms/hf_llm.py` | 54 | `torch.cuda.mem_get_info(f'cuda:{_device}')` called unconditionally when `use_gpu=True` — most likely runtime failure point | MEDIUM |

---

## Likely Fine (transparent under ROCm PyTorch)

These `torch.cuda.*` calls are remapped by the ROCm PyTorch wheel and require no code changes:

- `torch.cuda.empty_cache()` — `magellan/updater.py:244,326`, `lamorel/hf_llm.py`
- `torch.cuda.synchronize()` — `lamorel/hf_llm.py` (off by default in dev config)
- `torch.cuda.device_count()` — `lamorel/server/server.py`
- All `.to(self.device)` calls — already device-agnostic
- DDP distributed backend — already uses `gloo`, not `nccl`
- All `"cuda:N"` device strings — ROCm re-uses this namespace

---

## Unknowns / Needs Testing

- `torch.cuda.mem_get_info` behavior on multi-GPU AMD setups (needs live test)
- PEFT multi-adapter scheme on ROCm (no known issues, but untested)
- Whether RCCL is installed in the target environment
- `bitsandbytes` import side-effects at import time (peft ≥0.6 lazily imports it)

---

## Dependency Analysis

| Package | Status |
|---|---|
| **PyTorch** | Not pinned in `pyproject.toml` — swap in the ROCm wheel with no config changes |
| **bitsandbytes** | Primary blocker; ROCm support exists experimentally via `BNB_ROCM_ARCH` env var or `bitsandbytes-rocm` fork |
| **flash_attn** | Not used anywhere — no issue |
| **NCCL/RCCL** | Outer distributed group needs RCCL installed on the AMD system; Accelerate will auto-detect |
| **accelerate / peft** | No AMD-specific issues |

---

## Implementation Plan

Ordered by impact. Items 1 and 2 are the minimal set to get a first boot on AMD.

### 1. Make `bitsandbytes` optional (HIGH — unblocks all AMD runs)

- Move `bitsandbytes>=0.49.2` from main deps to an optional `[quantization]` extra in `pyproject.toml`
- Guard the `prepare_model_for_kbit_training` import in `magellan/initializer.py` behind `if use_4bit`
- This unblocks all runs where `load_in_4bit: false` (including the dev config)

### 2. Patch `hf_llm.py` in the Lamorel fork (MEDIUM — prevents crash on GPU memory query)

- Replace the `torch.cuda.mem_get_info(f'cuda:{_device}')` dict comprehension at line 54 with a try/except that falls back to `torch.cuda.get_device_properties(device).total_memory`
- Requires pushing a patch to the Lamorel fork and updating the dependency

### 3. Install ROCm PyTorch wheel (no code change)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

### 4. Ensure RCCL is installed

Install RCCL on the AMD system so `torch.distributed` can use it for multi-GPU communication. HuggingFace Accelerate will auto-detect it.

### 5. Set `HSA_OVERRIDE_GFX_VERSION` if needed

Some AMD GPU families require this env var to override the GFX version detection. Document the correct value for the target hardware in the setup notes.

### 6. Test path

Use the dev config (`load_in_4bit: false`) as the first target — it already avoids the bitsandbytes runtime path and is the lowest-risk starting point.

---

## Audit Notes

- Audited: `magellan/models.py`, `magellan/updater.py`, `magellan/initializer.py`, `magellan/main.py`, `magellan/goal_sampler.py`, all YAML configs, and all Lamorel source files
- No `flash_attn`, `xformers`, or NVTX usage found anywhere
- `pyproject.toml` does not pin PyTorch to a CUDA-specific wheel
