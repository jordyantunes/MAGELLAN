# FP32 Bottleneck Findings (RTX 4090, 2026-07-24)

Follow-up to `VRAM_TUNING.md` and `THROUGHPUT_SCALING.md`. Those docs got us to
`number_envs: 128` / batch 256/64 at ~8,000–8,500 ep/hr — and in doing so **moved the
bottleneck**. This doc records where the time goes now, what we changed about it, and the
microbenchmark evidence (`scripts/poc_speedups.py`) behind each change.

Hardware: 1× RTX 4090 (24 GB), 16 CPU cores (`nvidia-smi -L` — the two-GPU options in
THROUGHPUT_SCALING.md do not apply to this box; `n_llm_processes > 1` remains dead, see
the note in `local_gpu_config_magellan_4090_paper.yaml`).
Stack: torch 2.12.0+cu130, transformers 4.44.2, flan-t5-base fp32 + 4 LoRA adapters.

---

## Measured time budget (MLflow perf timers, run `7654c46a`, 128 envs, Docker)

Per ~9.6 s update cycle:

| Phase | s/cycle | Share |
|---|---|---|
| SAC update (2 grad steps) | 7.50 | ~78% |
| — `target_fwd` (score + critic_target, no-grad) | 2×1.118 | |
| — `policy_fwd` + `backward` (grad) | 2×0.914 | |
| — `critic_fwd` | 2×0.147 | |
| — unaccounted (~`policy_current_q_fwd` + `empty_cache` + tokenization; timers missing from stale Docker image) | ~3.1 | |
| LP recompute (31.3 s spike every `recompute_freq: 32` cycles, amortized) | 0.91 | ~10% |
| SR update | 0.60 | ~6% |
| Trajectory collection | 0.26 | ~3% |
| `test_policy` (amortized) | 0.35 | ~4% |

GPU util avg 83%; VRAM avg 6.3 GB / peak 11.1 GB of 24 GB.

**Conclusion:** the bottleneck is no longer CPU env collection (the 64-env story in
THROUGHPUT_SCALING.md) — it's GPU update math. And the whole pipeline ran **fp32 with
TF32 disabled** (grep-verified: no tf32/autocast/compile anywhere in `magellan/` or
lamorel; torch defaults TF32 off). Low VRAM usage is a *symptom* of
fp32-without-tensor-cores, not an invitation to bigger batches (VRAM_TUNING.md already
showed bigger batches are ~1.9× slower).

---

## PoC microbenchmark results (`scripts/poc_speedups.py`, GPU idle)

flan-t5-base + 4 LoRA adapters (r=16 α=32, `["q","v"]`), fp32, eval mode (matching the
real pipeline — lamorel never calls `.train()`, so HF gradient checkpointing is inactive
despite `PeftInitializer` enabling it):

| Benchmark (realistic SAC-update shapes) | fp32 | TF32 | Speedup |
|---|---|---|---|
| critic-style no-grad fwd (640 ctx → enc + 1-tok dec) | 595 ms | 475 ms | 1.25× |
| score-style no-grad fwd (64 enc → 640 dec, pre-encoded) | 168 ms | 122 ms | 1.37× |
| score-style fwd+bwd (LoRA grads) | 387 ms | 308 ms | 1.26× |

- **Mask build** (`models.py` per-token Python loop, 640×7): **53.2 ms vs 0.006 ms**
  vectorized (`output_tokens == pad`), bit-identical output. At ~17 score forwards per
  cycle (1 rollout + 8 target + 8 policy) that's **~0.9 s/cycle** of pure Python/kernel
  launch overhead.
- **`torch.cuda.empty_cache()`**: **42.3 ms/call** marginal cost → **~0.68 s/cycle** at
  16 calls/cycle (2 loops × 4 chunks × 2 updates).
- **LP-recompute chunk size** (2560 short prompts, TF32): 128 → 502 ms, 256 → 540 ms,
  512 → 591 ms, 1024 → 616 ms. **Bigger chunks are monotonically slower** on this
  GPU/model — the planned `minibatch_size: 1024` override was dropped.

TF32 note: 1.25–1.37× is real but below the theoretical tensor-core multiple —
flan-t5-base's 768-wide matmuls are small enough to be partly bandwidth/overhead-bound.

---

## Changes made (2026-07-24)

1. **TF32 enabled unconditionally** — `magellan/main.py`, module level before
   `lamorel_init()` so it runs in RL and LLM server processes. Precision: strictly finer
   than the 4-bit quantization the paper itself used; no fidelity concern.
2. **Vectorized pad mask** — `LogScoringModuleFn.forward` (`magellan/models.py`): the
   `mask[i, j] = False` double loop replaced by `mask = output_tokens == self._pad_token`
   (equivalence asserted in the PoC).
3. **`empty_cache_between_chunks` config flag, default off** —
   `magellan/updater.py` + `rl_script_args` in both 4090 configs. The calls were
   justified when peaks hit ~22 GB (VRAM_TUNING.md); at 9.6 GB peak they were pure sync
   overhead. Re-enable for VRAM-tight runs (e.g. stepping `delay_depth` toward 99, see
   PAPER_FIDELITY_PLAN.md).

Projected combined effect: ~0.9 + ~0.68 s/cycle removed + ~25% off the GPU-bound ~6 s
→ roughly **~3 s off a 9.6 s cycle (~30–45% more ep/hr)**. To be confirmed by A/B
(`check-speed` vs runs `7654c46a` / `832c0e23`).

**Smoke-run confirmation (2026-07-24, local 400-episode run, dev config):** clean exit,
zero errors, finite losses (policy ≈ −0.056, value ≈ 0.0007–0.004, α ≈ 0.049,
entropy ≈ 0.52). `perf/sac_update_seconds` avg **3.93 s/cycle over 35 cycles vs the 7.5 s
baseline — 1.9×** on the same 256-sample × 2-update workload. LP recompute spike 25.1 s
vs 31.3 s baseline (TF32 cut the GPU share; the CPU-side ~20 s remains, as predicted).

## Dropped after PoC

- **`lp_minibatch_size: 1024` for `compute_lp`** — measured *slower* (see sweep above).
  Also: projected GPU time for the full 25k×2 recompute is only ~10.5 s of the measured
  31.3 s — the remaining ~20 s is CPU-side (Python tokenization of 25k prompts in
  `hf_llm.forward` + gloo pickling of the goal strings each call), which chunk size
  cannot fix. Fixing that means caching tokenization in lamorel — see Deferred.

## Deferred (next rounds)

- **bf16 autocast on no-grad forwards** (lamorel fork `jordyantunes/lamorel` +
  `uv.lock` pin bump): the no-grad paths are the bulk of SAC-update GPU time; flan-t5 is
  bf16-native and the value/SR heads already upcast to fp32. Expected ~2× on those paths,
  on top of TF32.
- **CPU-side LP recompute cost** (~20 s/spike): tokenization caching for the fixed goal
  set, lamorel-side.
- **Critic restructure**: critic/critic_target/policy_current_q evaluate each
  (state, action) as a separate context — 640 encoder passes per chunk where the actor
  shares 64 via `pre_encode_inputs`. ~10× less critic encoder work, but changes the
  value-function architecture vs the paper implementation — deliberate-deviation
  experiment only.

## Housekeeping

- The Docker image used for run `7654c46a` is stale: it lacks the
  `policy_current_q_fwd` / `cuda_empty_cache` / `polyak_update` timers present at HEAD
  and logs `git_commit=unknown`. Rebuild before the A/B run.

## Update (2026-07-26): TF32 exonerated — the actual regression was elsewhere

A grasp-learning regression appeared shortly after this doc was written (no run since
~2026-07-20 learned `test/grasp`; see `REGRESSION_HANDOFF.md`). It was suspected to be
TF32, since that was this doc's only numerically-active change. Bisection proved
otherwise:

- A run with `NVIDIA_TF32_OVERRIDE=0` (driver-level TF32 kill switch) still got stuck —
  TF32 exonerated.
- Checking out commit `7001485` (immediately before the perf-instrumentation commit
  `4f33ebe`, i.e. before *this doc's* changes even existed) reproduced healthy learning.
- Checking out `4f33ebe` alone (before TF32/mask/empty_cache) reproduced the **stuck**
  signature — isolating the bug to that commit, not this one.

Root cause: `4f33ebe` changed `SACUpdater.update()`'s `sr_update` branch to return a
perf/GPU-mem dict, and `main.py` started gathering that return value over the same
distributed IPC channel (`dist.gather_object`) used by `sac_update`, corrupting the
actor/critic training path. Fixed by dropping that return value (see commit message on
the fix commit for detail). **This doc's projected gains stand** — TF32, the vectorized
pad mask, and `empty_cache_between_chunks` gating are all confirmed numerically inert
except for their intended perf effect; none contributed to the regression.
