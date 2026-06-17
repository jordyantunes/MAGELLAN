---
name: check-metrics
description: Fetch, interpret, and persist MLflow run metrics for a MAGELLAN training run. Accepts a run ID, MLflow UI URL, or "latest". Saves structured findings to .claude/findings/<run_id>.json and compares against previous runs.
tools: Bash, Read, Write
---

# check-metrics

Fetch, interpret, and persist MLflow run metrics for a MAGELLAN training run.

## Input

The argument is either:
- A MLflow UI URL: `http://localhost:5000/#/experiments/2/runs/<run_id>/...`
- A bare run ID: `5d2058d7f5b344deba1deea075b78bc1`
- The word `latest` to use the most recent run in the DB

## Steps

### 1. Extract the run ID

If the argument is a URL, parse the run ID from the path segment after `/runs/`.
If the argument is `latest`, run:
```bash
uv run .claude/skills/check-metrics/mlflow_metrics.py --list --experiment magellan
```
and pick the run with the most recent `start_time`.

### 2. Fetch metrics and params

```bash
uv run .claude/skills/check-metrics/mlflow_metrics.py <run_id> --summary
```

This returns a compact pre-digested JSON with params, final metrics, per-goal milestones (first episode each threshold was crossed), volatility stats, trajectory snapshots for stability metrics, and LP stats. It is safe for runs of any length — `--json` dumps full history and overflows the tool result buffer for runs longer than ~20k episodes.

As a side effect, the full per-metric history is written to `.claude/findings/<start_time_str>_<run_id>_history.json` (stderr note confirms the path). Load this file with `Read` when you need to investigate a specific curve in detail — e.g. a grow_plants collapse/recovery pattern — without re-fetching from MLflow.

**Volatility fields** (per goal metric, collapse threshold = 0.5):
- `max_value` / `step_of_max` — peak performance and when it occurred
- `min_after_first_crossing` — how far the metric fell after first crossing 0.5 (a value near 0 signals catastrophic forgetting)
- `n_collapses` — how many times it dropped back below 0.5 after crossing it
- `fraction_above_last_quarter` — fraction of final-25%-of-run points above 0.5 (1.0 = stably solved, <1.0 = still oscillating)

### 3. Interpret the results

Analyse the JSON output and produce findings across these dimensions:

**Policy learning**
- Did `test/grasp` reach ≥ 90%? By which episode?
- Did any `test/grow_*` metric exceed 5%?
- Check `volatility.<metric>.n_collapses` — any value > 0 means the policy forgot a goal after learning it. Check `min_after_first_crossing` to gauge severity (near 0 = catastrophic forgetting). Check `fraction_above_last_quarter` to see if it recovered stably.
- Is `train/entropy` finite and positive? A NaN value means padded action log-probs are unmasked (see `updater.py`). A value near 0 means the policy has collapsed to near-deterministic.
- Did `train/alpha` decay to near-zero (< 0.05)? If so, temperature collapsed — check if grasp is already saturated or if entropy NaN corrupted alpha tuning.

**SR head / LP signal**
- Are `diag/sr_impossibles` predictions near 0? If still ~0.34+ after many episodes, the SR head is not learning per-goal features.
- Compare `diag/raw_lp_*` (pre-threshold) vs `test/estimated_lp_*` (post-threshold). If raw LP is non-zero but estimated LP is zero, the 0.01 threshold in `goal_sampler.py:301` is suppressing real signal — consider lowering it.
- Did estimated LP spike and collapse? Spike → collapse on grasp is expected once grasp saturates. LP never rising on `grow_*` in a short run (<50k ep) is also expected.

**Training stability**
- Is `train/value_loss` decreasing and settling below 0.01?
- Is `train/policy_loss` trending negative (expected for SAC)?
- Any anomalies (sudden spikes, NaN)?

**Run config**
- Note key params: `goal_sampler`, `num_episodes`, `gradient_batch_size`, `buffer_size`, `seed`.

### 4. Compare against paper benchmarks

Consult `docs/paper_benchmarks.md` for extracted chart data (D.2 competence estimation, D.3 per-category SR, D.4 test-set generalization). Use the **Summary table** at the bottom of that file to classify each metric.

**Key checks:**
- Episode at which `test/grasp` and `test/grow_plants` cross 0.9 — compare to ON_TRACK range in table
- Final `test/grow_herbivores` and `test/grow_carnivores` vs. expected range for this run length (grow_carnivores is still converging at 500k in the paper — short runs should be labelled EXPECTED, not BELOW_PAPER)
- `diag/sr_impossibles` — should fall toward 0 by ~50k ep; >0.20 signals the SR head is not learning per-goal features
- Competence tracking on test goals: MAGELLAN should track observed SR within ±0.15; flat-zero estimate on grow_* indicates generalization failure (Online-ALP failure mode)
- LP signal: spike-then-collapse on grasp is expected once SR saturates; LP never rising on grow_* in runs <50k is also expected

Label each comparison as:
- `ABOVE_PAPER` — metric exceeds or reaches target faster than paper reports
- `ON_TRACK` — consistent with paper trajectory given run length
- `BELOW_PAPER` — underperforming vs. paper; note possible cause

Add these comparisons as findings with category `paper_comparison`.

### 5. Classify all issues

Label each finding as one of:
- `BUG` — confirmed code defect (e.g. entropy NaN, wrong mask)
- `EXPECTED` — known limitation of run length or task difficulty
- `INVESTIGATE` — ambiguous; needs another run or more data to confirm

### 6. Save findings

Findings files are named `<start_time_str>_<run_id>.json` where `start_time_str` comes from the `start_time_str` field in the `--summary` output (format: `YYYYMMDD_HHMMSS`, UTC). This makes files sort chronologically by experiment order.

Load the previous findings file if it exists (check `.claude/findings/` for a file ending in `_<run_id>.json`):

Write (or overwrite) it with this structure:
```json
{
  "run_id": "<run_id>",
  "recorded_at": "<ISO timestamp>",
  "params": { ... },
  "final_metrics": { ... },
  "summary": "<2-3 sentence plain-English summary of what this run showed>",
  "findings": [
    {
      "category": "policy_learning | sr_head | stability | config | paper_comparison",
      "severity": "BUG | EXPECTED | INVESTIGATE | ABOVE_PAPER | ON_TRACK | BELOW_PAPER",
      "metric": "<metric name or null>",
      "description": "<what was observed>",
      "recommendation": "<what to try next, or null>"
    }
  ],
  "comparison": "<if a previous findings file existed: 1-2 sentences on what changed vs that run, else null>",
  "paper_vs_run": "<1-2 sentences summarising how this run compares to the paper's reported results overall>"
}
```

Use `Write` to save the file. Then regenerate the changelog:

```bash
uv run .claude/skills/check-metrics/mlflow_metrics.py --changelog
```

This rewrites `.claude/findings/CHANGELOG.md` from all findings files in chronological order.

### 7. Report to user

Output:
1. One-paragraph plain-English interpretation of the run
2. Findings table (category | severity | metric | description)
3. Paper comparison: how key metrics compare to the paper's reported results (Figure 5, Table 2)
4. Path to the saved findings file
5. If a previous run was compared: what changed
