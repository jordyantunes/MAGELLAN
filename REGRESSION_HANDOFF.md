# HANDOFF: test/grasp success-rate regression (2026-07-25)

**RESOLVED 2026-07-26.** Root cause: `magellan/updater.py`'s `sr_update` branch (added in
commit `4f33ebe`) returned a perf/GPU-mem metrics dict, and `magellan/main.py` captured
and gathered that return value over the same distributed IPC channel (`dist.gather_object`)
used by `sac_update` — corrupting the actor/critic training path even though the SR loss
itself was never affected (which is why `sr_impossibles` looked healthy the whole time).
TF32, the vectorized pad mask, and `empty_cache_between_chunks` (all from `0caf5f2`, the
commit originally suspected) are fully exonerated. Fixed by dropping the `sr_update`
return value in both files. Verified: full `local_gpu_config_magellan_4090` config reaches
grasp=1.0 by ep 6007 — faster than any run in this project's history. See "Resolution"
section at the bottom for the full bisection chain. Everything below this point is the
original investigation, kept for the record.

---

For the next session picking this up. Status at handoff: **discriminating experiment running**
(run `d2796f7c`, "4090og_no_tf32") — see "Current test" below for the verdict procedure.

## The problem

Since ~2026-07-20, **no training run learns grasp**. `test/grasp` oscillates in the 0.0–0.16
band indefinitely instead of shooting to ~1.0 within a few thousand episodes. Confirmed
worst case: run `58cf3752` (paper config, seed 42) stuck ≤0.094 for **117k episodes**
(paper window for grasp≥0.9 is 20–50k; see `docs/paper_benchmarks.md` summary table).
Meanwhile throughput is excellent (~16k ep/hr on the 128-env config, ~18k ep/hr on the
paper config) after the FP32_BOTTLENECK.md perf changes — the runs are fast but not learning.

The SR head is NOT the problem: `sr_impossibles` now converges cleanly to ~0.002 (better
than the healthy-era runs, which had it stuck at 0.27). Estimated LP ≈ 0 everywhere is
*correct* behavior given the policy makes no progress. The regression is specific to the
actor/critic path.

## Evidence table (all runs seed 0 unless noted; grasp = `test/grasp`)

| Run | Started | Code era | Config | Result |
|---|---|---|---|---|
| `832c0e23` | 07-17 | pre-instrumentation image | 128 envs | **HEALTHY: grasp 0.5 @ ep 4014, 0.95 @ 5k, then 1.0 for 200k+** |
| `2a717e4c` | 07-17 | same era | 96 envs | HEALTHY: 0.5 @ ep 3006 |
| `7654c46a` | 07-20 | stale image (post-4f33ebe, **pre-TF32**) | 128 envs | 0.016 @ 3k (killed; healthy runs were 0.27–0.95 @ 3k) — weak evidence regression predates TF32 |
| `a9402350` | 07-24 | current code (TF32+mask+empty_cache) | 128 envs | **STUCK ≤0.16 @ 21k** — same config+seed as 832c0e23 ⇒ closest controlled A/B, implicates code not config |
| `6418601d` | 07-24 | current code | paper cfg (32 envs) | STUCK ≤0.17 @ 24k |
| `58cf3752` | 07-24 | current code | paper cfg, seed 42 | STUCK ≤0.094 @ 117k |
| `d2796f7c` | 07-25 | current code + `NVIDIA_TF32_OVERRIDE=0` | 128 envs | **RUNNING — the discriminating test** |

Healthy-run signature: grasp crosses 0.5 by ep 3–4k. Stuck signature: flat ≤0.16 forever;
entropy rises to ~1.5 then drifts to ~0.7 plateau; alpha decays to ~0.001 (alpha decay also
happened in healthy runs — not discriminating).

## Ruled out (verified this session)

- **Vectorized pad mask** (`models.py`, commit `0caf5f2`): bit-identical to the old loop —
  `torch.equal` assert in `scripts/poc_speedups.py` (`bench_mask`).
- **`empty_cache` removal** (`0caf5f2`): numerically inert (allocator behavior only).
- **Instrumentation commit `4f33ebe`**: full diff reviewed line-by-line — pure
  `perf.time(...)` wrapping plus timestamped output dirs / artifact logging. The
  `alpha_loss.backward()` reorder is inert (its graph only touches `log_alpha`; everything
  else is `.detach()`ed).
- **Config as cause**: `a9402350` reproduced the failure with the exact config+seed of the
  healthy `832c0e23`.
- **SR head / goal sampler**: healthy (see above); goals_distribution (80% impossibles) is
  identical in healthy runs.

## Suspects, ranked

1. **TF32** (`torch.backends.cuda.matmul.allow_tf32 = True` at top of `magellan/main.py`,
   commit `0caf5f2`) — the only numerically-active change in that commit. Being tested now.
2. **Unknown drift between the 07-17 and 07-20 Docker images** — supported weakly by
   `7654c46a` (pre-TF32 code, slow at 3k, but window too short to be conclusive). All runs
   log `git_commit=unknown` (no `.git` in image), so image contents can't be attributed to
   commits precisely. The 07-20 image also lacked some `4f33ebe` timers ⇒ it was built from
   an intermediate uncommitted state.

## Current test (verdict procedure)

Run `d2796f7c`, name `4090og_no_tf32`: 128-env config
(`local_gpu_config_magellan_4090`), seed 0, `output_dir outputs/magellan_tf32off`, launched
via `scripts/launch_run.sh` with `NVIDIA_TF32_OVERRIDE=0` (driver-level TF32 kill switch,
passed through `docker-compose.local.yml` environment — overrides the torch.backends flags,
no code change). At handoff: ~2k episodes, grasp 0.016 — too early to call
(healthy@2k was 0.109, crossing at 4k).

**Verdict at ~4–6k episodes** (`/check-metrics latest`, or grasp curve via MLflow):
- grasp > 0.5 by ~4–5k ⇒ **TF32 guilty.** Then: demote TF32 to a config flag default OFF
  (e.g. `rl_script_args.allow_tf32`, read at top of `magellan/main.py`), update
  `FP32_BOTTLENECK.md` (its projected gains stand, minus the TF32 share — mask +
  empty_cache alone were worth ~1.6s/cycle), rerun the paper experiment.
- grasp still ≤ ~0.1 at 6k ⇒ **TF32 exonerated.** Next bisect: `git checkout 7001485`
  (pre-instrumentation, the era of the healthy image), rebuild, run same config/seed
  ~5k episodes. If healthy there, walk forward (4f33ebe → 0caf5f2). If STILL stuck at
  7001485, the cause is outside git (base image, dependency resolution at build time —
  compare `uv.lock`-resolved versions inside old vs new images if both still exist:
  `docker images`).

Sanity check that the override took: `docker exec <magellan container> env | grep TF32`
should show `NVIDIA_TF32_OVERRIDE=0`.

## Tooling notes for the next session

- MLflow: the local venv's mlflow client cannot open `outputs/mlflow.db` directly (alembic
  schema `b7e4c1a90f23` newer than client). **Always use
  `MLFLOW_TRACKING_URI=http://localhost:5000`** (Docker mlflow server;
  `.claude/skills/check-speed/speed_metrics.py` honors this env var now).
- Findings JSONs: `.claude/findings/20260724_210555_58cf3752*.json` (the regression
  analysis), `.claude/findings/20260724_191855_6418601d*.json` (first INVESTIGATE flag).
- Perf/context docs: `FP32_BOTTLENECK.md` (why TF32 etc. were added),
  `THROUGHPUT_SCALING.md`, `VRAM_TUNING.md`, `PAPER_FIDELITY_PLAN.md`.
- Launch: `scripts/launch_run.sh` (interactive; env prefixes like
  `NVIDIA_TF32_OVERRIDE=0` pass through to the container).
- The stuck runs `6418601d` / `58cf3752` show status RUNNING in MLflow but their
  containers were stopped (`docker compose down` kills without ending the mlflow run).

## Do-not-forget

- The paper experiment (delay_depth staging 24→49→99, `PAPER_FIDELITY_PLAN.md`) is
  **blocked** until this regression is resolved; any results from the stuck runs are void.
- When resolved, update `FP32_BOTTLENECK.md` with the outcome so its recommendations
  don't mislead (it currently presents TF32 as a pure win).
- Uncommitted work on `experiments` branch: run_name feature (main.py + 8 configs +
  compose), `scripts/launch_run.sh`, compose TF32 passthrough, check-speed URI fix,
  findings files. User prefers separate, logically-scoped commits (see CLAUDE.md Commits).

## Resolution (2026-07-26)

Continued the "Current test" verdict from handoff: `d2796f7c` (TF32 off) ran to 106k
episodes still stuck ≤0.16 — **TF32 exonerated**. Followed the "walk forward" bisection
plan:

1. Checked out `7001485` (pre-instrumentation, immediately before `4f33ebe`), rebuilt,
   ran the same config+seed as the stuck runs: **healthy** — grasp 1.0 by ep 15017.
   Confirms the bug is in `4f33ebe` or `0caf5f2`, not base-image/dependency drift.
2. Checked out `4f33ebe` alone (perf instrumentation only, no TF32/mask/empty_cache):
   **stuck** — grasp flat ≤0.11 through 55k episodes. Isolates the bug to `4f33ebe`
   specifically; `0caf5f2` fully exonerated.
3. Re-diffed `4f33ebe` against `7001485` line-by-line (the original review only checked
   the `perf.time()` wrapping and the `alpha_loss.backward()` reorder, both genuinely
   inert). Found two *new* side-effecting additions in `main.py` that weren't part of
   that review: `mlflow.log_artifacts` on checkpoint save, and — the actual culprit —
   `sr_update`'s return value being captured and gathered.
4. Reverted only the `sr_update` return-value change on top of unmodified `4f33ebe`:
   **healthy** — grasp 1.0 by ep 10007 over a bounded 20k-episode run. Root cause
   confirmed.
5. Applied the same revert to `magellan/main.py` / `magellan/updater.py` at `experiments`
   HEAD, rebuilt `magellan:latest`, verified with the real `local_gpu_config_magellan_4090`
   config: grasp 1.0 by ep 6007 over 20k episodes — the fastest healthy run on record.

**Root cause:** `SACUpdater.update()`'s `sr_update` branch (in `magellan/updater.py`)
went from an implicit `return None` to `return {**perf.as_metrics(...),
**gpu_mem_snapshot(...)}`. `magellan/main.py` then captured that into `sr_update_results`
and gathered/logged it. This call goes through the same distributed IPC path
(`Caller.update` → `dist.gather_object` across the RL process and LLM worker process) as
`sac_update`. Introducing a gathered return payload on a call that previously had none
perturbed that shared channel enough to corrupt the actor/critic training — despite the
SR loss computation itself being fine the whole time (which is why `sr_impossibles`
never flagged anything).

Findings: `.claude/findings/20260725_133732_d2796f7cc45b4bf6b5e9a6c36a68b92e.json` (TF32
exoneration), `..._f194847f...json` (7001485 healthy), `..._fc48b71d...json` (4f33ebe
stuck), `..._6c532ea9...json` (bisect fix confirmation), `..._1b6b62b5...json` (real-branch
fix verification).

Paper-fidelity experiment plan is now unblocked.
