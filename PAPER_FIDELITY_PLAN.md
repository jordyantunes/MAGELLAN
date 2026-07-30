# Plan: Paper-Faithful 4090 Config

Goal: create a run config that matches the MAGELLAN paper's hyperparameters
(Appendix C.1 Table 5, C.2 Table 6, C.4 compute budget) as closely as feasible
on a single 24GB RTX 4090, starting from `local_gpu_config_magellan_4090.yaml`
(the throughput-tuned config) and the original `local_gpu_config_magellan.yaml`.

New config: `configs/little_zoo/local_gpu_config_magellan_4090_paper.yaml`

## Changes made vs. `local_gpu_config_magellan_4090.yaml`

| Key | 4090 (throughput) | New (paper) | Why |
|---|---|---|---|
| `warmup_updates` | 5 | 10 | Table 5 / text: "warm-up phase of 10 updates" (Q-function only) |
| `buffer_size` (SAC) | 250000 | 500000 | Table 5: "Replay buffer capacity 500000" |
| `magellan_args.delay_depth` | 6 | 24 (staged toward 99) | See finding below — this is what actually sets the paper's "ℬ size: 100", not `N` |

## Kept as deliberate deviations (not changed)

- `n_llm_processes: 1`, `load_in_4bit: false` — paper used 2×H100 80GB with 4-bit
  quantization; 4-bit crashes on this stack (transformers==4.44.2 + explicit
  device_map). Not fixable via config.
- `number_envs: 128` / `update_freq: 128` kept at paper's `32`/`64` in the new
  config for fidelity — NOTE this reverts the THROUGHPUT_SCALING.md-validated
  speedup. It's the correct call for an apples-to-apples paper comparison, but
  expect substantially lower episodes/hr than the throughput-tuned config.
- `minibatch_size` / `gradient_batch_size` (256/64) — VRAM-tuned lamorel
  batching knobs, not paper table values (paper doesn't pin these).

## Non-obvious code finding: `magellan_args.N` is dead

Table 6 lists "ℬ size: 100" — the number of past SR-estimator weight snapshots
kept for computing learning progress. The config has an `N: 100` field that
*looks* like it sets this, but tracing the code:

- `goal_sampler.py:241` — `self.N = magellan_args.N` is assigned and never
  read again anywhere in the codebase.
- `goal_sampler.py:253` — `self._buff_size = magellan_args.delay_depth + 1` is
  the value actually passed to `SACUpdater.weights_buffer` (a `deque(maxlen=...)`
  of cloned LoRA+MLP weights, see `updater.py:362-371`).

So the real paper-equivalent value for `delay_depth` is **99** (buffer=100),
not the `3`/`6` used in the existing configs — both prior configs have a much
narrower LP window than the paper (`|B| × recompute_freq` = 100×32=3200
episodes in the paper vs. 128/224 in the existing configs).

`weights_buffer` entries are live tensor clones on whatever device the LoRA
params live on (GPU), captured every `recompute_freq` (32) updates — so
raising `delay_depth` to 99 adds real, so-far-unmeasured VRAM overhead on top
of an already-tight 24GB budget (VRAM_TUNING.md: 4090 runs at ~2.3-2.8GB free
at peak with `gradient_batch_size: 128`, and the paper-fidelity config uses
smaller batches so has more headroom, but this is untested).

## Next steps

1. ~~Launch a run with the new config, `delay_depth: 24` as a first step up
   from 6, and watch `nvidia-smi` / OOM behavior.~~ Done — run
   `57bc170284db4035ba6bb07b9c68f47c`. Confirmed params match
   (`number_envs=32`, `minibatch_size=256`, `gradient_batch_size=64`,
   `buffer_size=500000`), preliminary throughput ~7-8k ep/hr (early window,
   not steady-state — episode length grows as the agent learns). GPU trace
   (`gpu_report_4090_paper.csv`) showed peak VRAM only ~10.5/24GB, avg util
   72% — much more headroom than the throughput-tuned config's ~18-22GB
   peaks.
2. ~~Set `n_llm_processes: 2` given the VRAM headroom.~~ **Confirmed dead end**
   — crashes with `IndexError: list index out of range` in
   `lamorel/server/llms/base_llm.py:10` (`self.device_id = self.devices[0]`).
   Root cause: `lamorel/server/server.py:88-90` splits `torch.cuda.device_count()`
   GPUs across LLM processes via `np.array_split` with no overlap — on a
   1-GPU box the 2nd LLM process gets `devices=[]`. This isn't a VRAM/tuning
   problem despite the headroom we measured; it's architectural. Reverted to
   `n_llm_processes: 1` with a comment explaining why. `n_llm_processes: 2`
   requires ≥2 physical GPUs and is not achievable on this single-4090 setup —
   not worth retrying here.
3. Step `delay_depth` up (49, then 99) and re-check VRAM each time, rather
   than jumping straight to 99.
4. Compare episodes/hr and MLflow metrics (SR curves, LP estimation error)
   against both the throughput-tuned run and, if possible, paper Figure 5/
   Table 2 numbers.
5. Once a safe max `delay_depth` is found, record it in VRAM_TUNING.md the
   same way the `minibatch_size`/`gradient_batch_size` findings are recorded.
