# LP Collapse Investigation

## Background

A 10k-episode dev run on the RTX 5080 (MAGELLAN goal sampler, `local_gpu_config_magellan_dev`) revealed three issues:

- `train/entropy` logged as `nan` throughout training
- `estimated_lp_*` collapsed to 0 for all goal categories by episode ~3000, despite `grow_*` goals remaining unsolved
- The SR head predicted ~0.34 for **all** categories including impossibles (true SR should be ~0 for impossibles)

---

## Fix 1 — Entropy NaN (`magellan/updater.py`)

**Problem:** Entropy was computed without masking `-inf` log-probs from padded actions:

```python
# broken
entropy_log.append(torch.mean(-torch.sum(action_probs * action_log_probs, dim=-1)))
```

`action_log_probs` contains `-inf` for padded (invalid) actions. `action_probs` for those entries rounds to 0.0 in float32, so `0.0 * -inf = NaN`. This produced `train/entropy: nan` in every logged step.

**Impact:** Logging only — entropy is not used in any backward pass and does not affect policy or alpha training.

**Fix:** Apply the same `masked_fill` used on the loss terms (mask is already in scope from line 221):

```python
# fixed
entropy_log.append(torch.mean(-torch.sum(action_probs * action_log_probs.masked_fill(~mask, 0.0), dim=-1)))
```

---

## Fix 2 — Diagnostic raw LP logging (`magellan/main.py`)

**Problem:** The only LP metrics logged (`estimated_lp_*`) are computed **after** the `lp[lp < 0.01] = 0.0` numerical stability threshold in `goal_sampler.py`. If the raw `|sr_current - sr_delayed|` difference is small but real (e.g. 0.005), the threshold kills it and it's invisible in MLflow.

**Fix:** Inside the `test_freq` evaluation block, add a `diag/` metric group that logs raw LP before the threshold, using a fixed sample of 8 goals per category:

```python
if use_magellan and is_rl_process:
    if not hasattr(goal_sampler, '_diag_goals'):
        goal_sampler._diag_goals = {
            c: [g for g in test_goals['grasp'] + test_goals['grow_plants'] + ...][:8]
            for c in ['grasp', 'grow_plants', 'grow_herbivores', 'grow_carnivores', 'impossibles']
        }
    diag = {}
    for cat, gs in goal_sampler._diag_goals.items():
        if gs:
            sr, sr_d, _ = goal_sampler.compute_lp(gs)
            diag[f"diag/sr_{cat}"] = float(sr.mean())
            diag[f"diag/sr_delayed_{cat}"] = float(sr_d.mean())
            diag[f"diag/raw_lp_{cat}"] = float(np.abs(sr - sr_d).mean())
    mlflow.log_metrics(diag, step=ep)
```

Reuses `MAGELLANGoalSampler.compute_lp()` — no new code paths. Only runs on the RL process.

**How to interpret:**
- `diag/raw_lp_*` > 0 but `estimated_lp_*` = 0 → threshold is masking real signal; lower the 0.01 threshold
- `diag/raw_lp_*` ≈ 0 → SR head genuinely isn't detecting change; delay window too short or head not learning
- `diag/sr_impossibles` ≈ 0.5 after many episodes → SR head not converging; check BCE loss and LR

---

## Fix 3 — Configurable delay depth (`magellan/goal_sampler.py`, configs)

**Problem:** The delayed-weight buffer maxlen was hardcoded as `int(N / recompute_freq + 1) = 4`. Once full, `set_weights(idx=0)` looks back exactly 3 snapshots × 32 sampler steps = 96 steps. For a 500k-episode production run this is a reasonable delay. For a 10k dev run (~156 total update cycles), a 96-step delay means the "delayed" SR is comparing against weights from 60%+ into the run's history — too coarse to be useful as a diagnostic.

**Fix:** Read `delay_depth` from `magellan_args` (with default 3 to preserve current behaviour) and store `buff_size = delay_depth + 1` on the sampler instance:

```python
# goal_sampler.py — MAGELLANGoalSampler.__init__
self._buff_size = getattr(magellan_args, 'delay_depth', 3) + 1
self.agent.update([""] * 8, [[""]] * 8, func='update_buffer', buff_size=self._buff_size)

# goal_sampler.py — MAGELLANGoalSampler.update
self.agent.update([""] * 8, [[""]] * 8, func='update_buffer', buff_size=self._buff_size)
```

**Config values:**

| Config | `delay_depth` | Buffer maxlen | Effective delay |
|--------|--------------|---------------|-----------------|
| `local_gpu_config_magellan.yaml` | 3 | 4 | 96 sampler steps (unchanged) |
| `local_gpu_config_magellan_dev.yaml` | 6 | 7 | 192 sampler steps |

---

## Files changed

| File | Change |
|------|--------|
| `magellan/updater.py` | Mask `action_log_probs` before entropy sum |
| `magellan/main.py` | Add `diag/` raw LP metrics inside `test_freq` block |
| `magellan/goal_sampler.py` | Use `delay_depth` from config for `_buff_size` |
| `configs/little_zoo/local_gpu_config_magellan.yaml` | Add `magellan_args.delay_depth: 3` |
| `configs/little_zoo/local_gpu_config_magellan_dev.yaml` | Add `magellan_args.delay_depth: 6` |

---

## How to re-run and verify

```bash
./run_magellan.sh --dev --seed 0
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5000
```

Check in the new run:
1. `train/entropy` — should be a positive finite value, trending down as grasp policy becomes confident
2. `diag/raw_lp_grasp` vs `estimated_lp_grasp` — compare to see if the threshold is suppressing signal
3. `diag/raw_lp_grow_*` — near zero throughout confirms the run is too short, not a bug
4. `diag/sr_impossibles` — should trend toward 0 if the SR head is learning; if stuck at ~0.5, investigate SR optimizer LR
