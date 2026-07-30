---
name: check-speed
description: Measure MAGELLAN training throughput (episodes/hour) for a run, optionally cross-referenced with an nvidia-smi GPU CSV trace, and compare against previously recorded runs. Accepts a run ID, MLflow URL, or "latest".
tools: Bash, Read
---

# check-speed

Measure training throughput for a MAGELLAN run and compare it against past runs — the same
episodes/hour + GPU-utilization analysis used when tuning `THROUGHPUT_SCALING.md` /
`VRAM_TUNING.md`.

## Input

The argument is either:
- A bare run ID: `2a717e4c09de4190b843ab55252a22da`
- An MLflow UI URL (run ID is parsed out)
- The word `latest` to use the most recently started run
- Optionally, a path to an `nvidia-smi --query-gpu=... --format=csv` trace to fold into the report

## Steps

### 1. Measure throughput for the target run

```bash
uv run .claude/skills/check-speed/speed_metrics.py <run_id_or_latest> [--gpu-csv path/to/report.csv]
```

This computes episodes/hour from the first→last logged point of `test/grasp` (step count doubles
as episode count, timestamp gives wall-clock elapsed time) — the same method used to derive every
ep/hr figure in `THROUGHPUT_SCALING.md`/`VRAM_TUNING.md`. It also pulls the throughput-relevant
params (`number_envs`, `update_freq`, `minibatch_size`, `gradient_batch_size`, `seed`,
`goal_sampler`).

If `--gpu-csv` is given, it summarizes that trace: average/min/max GPU utilization, average/min/max
VRAM used, and the fraction of samples at 0% or <50% utilization (a proxy for CPU-bound stalls —
see the "troughs" discussion in `VRAM_TUNING.md`).

A findings file is saved automatically to `.claude/findings/speed/<start_time>_<run_id>.json`
unless `--no-save` is passed. Re-running on the same run_id later (e.g. to check if the rate held
up after more episodes) overwrites that file with the newer numbers — mention in your report if
the number moved meaningfully from a prior check.

### 2. Sanity-check the read

- If `speed` comes back `null`/`n/a`, the run has fewer than 2 `test/grasp` points yet (`test_freq`
  episodes haven't elapsed) — say so rather than reporting a rate.
- A rate that looks abnormally low can mean the run is CPU-bound (check the GPU trace, if given,
  for frequent low-utilization samples) — or, if it looks like the episode counter isn't moving at
  all, check for the `update_freq < number_envs` bug documented in `THROUGHPUT_SCALING.md`
  (integer-division-to-zero in `collect_trajectories`, `magellan/main.py:61` — silently spins
  forever collecting nothing, no crash).

### 3. Compare against prior runs

```bash
uv run .claude/skills/check-speed/speed_metrics.py --compare
```

Prints every previously saved speed finding, fastest first, with its throughput params side by
side. Use this to state where the current run lands relative to history (e.g. "+38% over the best
64-env run").

To see live throughput for *any* run without saving a comparison record (e.g. a broad initial
scan):

```bash
uv run .claude/skills/check-speed/speed_metrics.py --list
```

### 4. Report to the user

1. The measured ep/hr for the target run, with episode range and elapsed hours it was computed
   over (small windows — a handful of data points — are a preliminary read, say so).
2. GPU utilization/VRAM summary, if a CSV was provided.
3. A comparison table against the fastest/most relevant prior runs from `--compare`, stating the
   %  difference.
4. Path to the saved findings file.
5. If this same run_id was checked before, note whether the rate changed meaningfully between
   checks (stable rate over a longer window is stronger evidence than a single early reading).

### 5. Keep the docs in sync (only if asked)

If the user asks to update the comparison docs, fold new numbers into `THROUGHPUT_SCALING.md`'s
"Measured result" section and `VRAM_TUNING.md`'s episodes/hr table — don't do this automatically
on every check, only when told to.
