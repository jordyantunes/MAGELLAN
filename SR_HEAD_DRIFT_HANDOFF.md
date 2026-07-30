# HANDOFF: SR head competence-estimation drift (2026-07-28)

**Status: UNRESOLVED, root-cause investigation in progress.** This is a distinct bug from
the (resolved) grasp regression in `REGRESSION_HANDOFF.md` — read that file only for
historical context; it is not related to this issue.

## UPDATE (2026-07-28, continued session): two new confirmed findings

1. **Reward signal confirmed clean binary — lead #1 from "Next steps" below is ruled
   out.** `little_zoo/playground/reward_function.py:get_reward_from_state` returns a plain
   Python `bool`; `littlezoo.py:132` returns `float(goal_reached)` and the episode
   terminates (`done = truncated or goal_reached`) the same step the goal is reached, so a
   single episode's `ep_ret` can only ever be exactly `0.0` or `1.0`. No partial credit, no
   shaping. `success_buffer` labels are clean.

2. **Buffer category-imbalance ruled out — lead #3 (goal_buffer category mix) is empirically
   healthy.** Loaded `goal_buffer.pkl`/`success_buffer.pkl` from the live run's ep-263886
   checkpoint (`outputs/magellan_paper_delay99/2026-07-27-20-22-58/263886/`) and matched
   each entry against the reconstructed train goal-category sets (`generate_goals(seed=0,
   [20000,4000,800,160,32])`). Raw composition: **impossibles 84.7%** (label 0.0 for all),
   grasp 8.5% (label 1.0), grow_plants 1.1% (label 1.0), grow_herbivores 0.7% (label 0.0),
   grow_carnivores 0.1% (label 0.0). Re-simulated the actual recency-weighted `p = i/Σj`
   sampling formula (256-batch × 50 draws) — weighted composition is nearly identical to
   raw (85.5% impossibles, still all label 0.0); mass in the most-recent 1000 of 5000
   buffer slots is only 36%, not enough to erase the impossible-heavy mix. **The SR head is
   seeing overwhelmingly-correct negative labels for impossible goals throughout training,
   yet still drifts its predictions upward on them — this is not a data/label problem.**
   (The `i/Σj` recency-weighted sampling formula itself was also checked against the paper:
   it matches Appendix C.2 exactly, so it is correct by design, not a bug.)

3. **CONFIRMED BUG (real, but not yet proven to be the drift's root cause): SR-update
   gradient-accumulation step count uses the wrong batch-size config.** In
   `magellan/updater.py:115` (`sr_update` branch):
   ```python
   gradient_accumulation_steps = math.ceil(self._minibatch_size / self._gradient_batch_size)
   ```
   `self._minibatch_size` is `rl_script_args.minibatch_size` (SAC's replay minibatch size),
   set in `SACUpdater.__init__` (`main.py:255`) — **not** `magellan_args.batch_size`, the
   actual length of the `goals`/`success` lists this call trains on (sampled at
   `main.py:451`, size `magellan_args.batch_size`). These are two independent config knobs:

   | Config | `rl_script_args.minibatch_size` | `gradient_batch_size` | `magellan_args.batch_size` | accumulation_steps computed | real non-empty chunks |
   |---|---|---|---|---|---|
   | base (`local_gpu_config_magellan.yaml`) | 512 | 192 | 256 | ceil(512/192) = **3** | 2 (192+64) |
   | paper (`local_gpu_config_magellan_4090_paper.yaml`) | 256 | 64 | 256 | ceil(256/64) = **4** | 4 (64×4) |

   In the base config, `goals[256:384]` and beyond (chunks 2+) are empty slices on a
   256-element list, silently skipped via `if _batch_size <= 1: continue` — so only 2 real
   chunks contribute gradient, but `loss = sr_loss / gradient_accumulation_steps` divides by
   3, scaling the effective SR gradient to ~2/3 strength. In the paper config the two
   batch-size knobs coincidentally match, so `gradient_accumulation_steps` correctly equals
   4 real chunks — no bug there. **This plausibly explains the previously-unexplained
   timing differential** (paper config drifts faster/worse at ep 105-135k vs base config's
   ep 255-270k): the paper config trains the SR head at full strength while the base config
   is silently under-trained by this bug, so the paper config reaches whatever bad fixed
   point causes the drift sooner. **This does not yet explain the drift's root cause** —
   only why one config manifests it faster than the other. Fix: compute
   `gradient_accumulation_steps` from `len(kwargs['goals'])` (or pass `magellan_args.batch_size`
   into `SACUpdater`), not from `self._minibatch_size`, when `func == 'sr_update'`.

**Where this leaves the investigation:** the two cleanest hypotheses from the original
handoff (dirty reward signal, category-imbalanced training data) are both ruled out with
direct evidence. The confirmed gradient-accumulation bug is real and worth fixing, but by
itself doesn't explain an *upward* drift specifically on negative-labeled/rare categories —
it only explains differential onset speed between the two configs. The root cause of the
drift direction itself is still open. Remaining unstarted leads from the original list:
#4 (verify the `set_weights` positional zip — requires spinning up the actual LLM+LoRA
stack, not done this session) and #5 (log raw pre-sigmoid SR logit mean/std over training
to see if the SR head is collapsing toward a dominant bias term rather than genuinely
confusing categories — this is now the most promising remaining lead, since a shared-bias
collapse would be consistent with all three categories drifting in lockstep despite correct
label proportions).

## The problem

MAGELLAN's SR head (the LLM-based competence/success-rate estimator that drives the
learning-progress goal sampler) starts out calibrating correctly early in training, then
drifts badly out of calibration as training progresses: `test/estimated_sr_impossibles`
(and, in lockstep, `test/estimated_sr_grow_herbivores` / `test/estimated_sr_grow_carnivores`)
climb from a healthy ~0.15-0.19 trough up to 0.6-0.85, even though:

- Impossible goals are by construction unwinnable (true SR must be ~0).
- `test/grow_herbivores` and `test/grow_carnivores` (true SR, measured by actually playing
  episodes) stay at 0.0 the entire time in both runs studied so far.

Meanwhile `test/grasp` and `test/grow_plants` (the actor/critic side) learn perfectly
cleanly in both runs — this is purely an SR-head / competence-estimation bug, not a
recurrence of the resolved actor/critic regression.

This matters because MAGELLAN's whole sampling mechanism depends on
`LP = |sr_current - sr_delayed|` being a meaningful signal. If the SR head can't
distinguish impossible goals from goals that are simply hard/unsolved-so-far, the goal
sampler's LP-driven budget allocation degrades toward noise — and there is now direct
evidence this is happening: `test/grow_herbivores`/`grow_carnivores` have missed the
paper's expected onset window (150-200k episodes) in the most recent run.

## Evidence: reproduced across two independent configs

| Run | Config | `minibatch_size` | `delay_depth` | Drift onset (~0.6) | Peak / last seen |
|---|---|---|---|---|---|
| `5c5241580a424056b0d4516032dfcb6d` | base `local_gpu_config_magellan.yaml` | 512 | 3 | ep 255-270k | 0.57-0.74 @ ep 365k |
| `9bc9ddc8a79d4568999c5560594fc154` | paper-fidelity `local_gpu_config_magellan_4090_paper.yaml` | 256 (paper-correct) | 99 (paper-correct) | ep 105-135k (**faster**) | peaked 0.78-0.85 @ ep 180-220k, settled noisy 0.52-0.76, last 0.7558 @ ep 263k |

Both runs: seed 0, `google/flan-t5-base`, `goals_distribution=[20000,4000,800,160,32]`.

**Ruled out:**
- **Batch size / delay_depth as cause** — the corrected config (256/99, matching paper
  Table 5/6 exactly) drifts *faster and worse* than the wrong config (512/3), not better.
  If either param were the cause, fixing them should have helped.
- **Seed noise** — two pre-fix runs on the exact same 32-env/500k-buffer paper config
  (`6418601daeec4` seed 0, `58cf3752c49b4` seed 42) show `estimated_sr_impossibles`
  converging cleanly to ~0.0002-0.002 and staying there. But those runs are from *before*
  the grasp-regression fix, when grasp/grow_plants never learned — nearly every episode
  across every category ended in failure, so "predict success ≈ 0 everywhere" was
  trivially correct and doesn't actually test discrimination. Not a clean comparison.
- **VRAM/OOM/instability** — both runs have healthy `train/value_loss`, `train/policy_loss`,
  `train/entropy` (finite, no NaN), and `train/alpha` (decays as expected). VRAM stable at
  ~12GB/24GB on the delay_depth=99 run through 23h+.

**Leading (unconfirmed) hypothesis going in:** the SR head faces a genuinely harder
discrimination task once grasp/grow_plants start succeeding at high rates (timing
correlates: drift accelerates right as grow_plants ramps through its solve window in both
runs) — but this is speculative and was not confirmed before the investigation below
started finding more concrete leads.

## Code walked through this session (`magellan/` — read these before re-deriving)

- **`models.py`**: `SRHeadModuleFn` (~line 110). MLP `hidden→128→1`, Tanh activation,
  returns a raw **logit** (no sigmoid in `forward()` — confirmed correct, sigmoid is
  applied later in `goal_sampler.py`). Two instances are created in `main.py` (~line 232):
  `'sr'` (trainable, adapter = `magellan_args.sr_adapters`) and `'delayed'` (frozen,
  hardcoded to filter on `.delayed_adapters.` regardless of the `adapters=` ctor arg it's
  given — the ctor arg is effectively unused dead weight for the delayed branch, not a bug
  but confusing naming).
- **`updater.py`**:
  - `sr_update` branch (~line 92-153): trains the `'sr'` head+adapter via
    `F.binary_cross_entropy_with_logits(sr, _success)` where `_success` comes from
    `kwargs['success']` — traced this back to `main.py`'s `success_buffer` (see below).
    Confirmed no return-value/IPC issue here (that was the already-resolved, unrelated bug).
  - `update_buffer`/`set_weights` (~line 391-405): `update_buffer` snapshots the current
    `'sr'` adapter+MLP weights (`self._sr_parameters_filter`) into a
    `deque(maxlen=delay_depth+1)` every `recompute_freq` calls. `set_weights(idx=0)` copies
    `weights_buffer[0]` (the oldest entry currently in the deque — correctly represents
    "delayed" once the deque is full) into the `'delayed'` head+adapter's parameters.
  - **Untested/fragile spot**: `set_weights` does
    `zip(self._iterator_named_filtered_params(self._sr_delayed_parameters_filter), self.weights_buffer[idx])`
    — this zips two *separately filtered* parameter iterators (one selecting
    `.sr.`/`.sr_adapters.` params, the other selecting `.delayed.`/`.delayed_adapters.`
    params) purely by **position**, not by matching parameter name. This only works if
    both filters walk the model's `named_parameters()` in exactly parallel order (i.e. each
    LoRA-targeted layer emits its `sr_adapters` and `delayed_adapters` weights at
    structurally-corresponding positions). This is plausible-by-construction (PEFT should
    register adapters in a consistent per-layer order) but **was not empirically verified
    this session** — worth an explicit assertion/check (e.g. compare parameter shapes and
    counts pairwise, or temporarily zip with names printed) before ruling it out.
- **`initializer.py`**: `PeftInitializer.initialize_model` — `default`, `delayed_adapters`,
  `sr_adapters`, `critic_target` are each added via `peft_model.add_adapter(name, config)`,
  each getting an **independent random LoRA init** (standard PEFT: `A` random, `B` zero →
  zero delta at t=0 regardless of `A`, so all adapters start numerically identical to the
  frozen base). `delayed_adapters` and `critic_target` explicitly frozen
  (`requires_grad=False`); `default` and `sr_adapters` explicitly trainable. Consistent with
  observed ep-0 baseline (~0.52, i.e. ≈ sigmoid(0) plus MLP-head randomness) matching across
  `sr`/`sr_delayed` at the very first `compute_lp` call.
- **`goal_sampler.py`**: `MAGELLANGoalSampler.compute_lp` (~line 287) — calls
  `set_weights(idx=0)` then runs the `'delayed'` and `'sr'`/`'value'` module functions,
  applies `F.sigmoid` to both, takes `lp = |sr - sr_delayed|`, and zeroes `lp < 0.01`
  (numerical-stability threshold, not a suspect here). `self.N` (`magellan_args.N`) is
  confirmed-dead code (see `project-delay-depth-vs-n` memory) — `_buff_size` is what
  actually matters and is `delay_depth + 1`.
- **`main.py`** (~line 393-395, 447-470): **This is the active lead, interrupted
  mid-investigation:**
  ```python
  if config_args.rl_script_args.goal_sampler == "magellan":
      goal_buffer.extend(data['goals'])
      success_buffer.extend(data['ep_ret'])
  ...
  success = [success_buffer[i] for i in idx]
  ...
  agent.update(..., goals=goals, success=success, func='sr_update', ...)
  ```
  The SR head's BCE training target is **`data['ep_ret']`** — the raw episode return
  accumulated in `collect_trajectories` (`ep_ret[i] += rewards[i]` per step, appended to
  `data["ep_ret"]` on episode end) — **not an explicitly separate binary success flag.**
  This is the single most promising unexplored lead: if `ep_ret` is not cleanly binary
  {0.0, 1.0} per episode, BCE-with-logits training on it will not correctly calibrate a
  success *probability*, and could plausibly produce exactly the observed symptom (all
  categories' predicted "success" drifting upward together as overall reward-shaping
  signals change with policy behavior, even for goals that never actually succeed).

  **Traced so far, NOT fully confirmed:** `little_zoo/littlezoo.py:119`
  (`.venv/lib/python3.12/site-packages/little_zoo/littlezoo.py`) computes
  `goal_reached = get_reward_from_state(o, self.env_desc[0], self.env_params)` and returns
  `float(goal_reached)` as the per-step reward; the episode terminates immediately on
  `done = truncated or goal_reached` (line 122). At face value this means an episode's
  cumulative `ep_ret` should be exactly 0.0 (failure/truncation) or exactly 1.0 (success,
  terminal step reward once, episode ends same step) — i.e. binary by construction. **But
  this was not verified all the way down**: the investigation was about to open
  `little_zoo/playground/reward_function.py`'s `get_reward_from_state` (the actual
  boolean/value-returning function) to confirm it truly returns a clean 0/1 (or bool) and
  not some partial-credit/shaped value, when the session was redirected to write this
  handoff instead. **This is the very next thing to check.**

## Next steps for the next session (in priority order)

1. **Finish the interrupted lead**: read
   `.venv/lib/python3.12/site-packages/little_zoo/playground/reward_function.py`
   (`get_reward_from_state`) and confirm its return type/range. If it's not a clean binary
   0/1 per episode, that's very likely the root cause — the fix would be to derive an
   explicit binary success label at the `collect_trajectories` call site in `main.py`
   (e.g. `rewards[i] > 0` or equivalent) rather than feeding `ep_ret` directly into
   `success_buffer`.
2. **If reward is confirmed clean binary**, check for a **data-alignment bug** instead:
   verify `data['goals']` and `data['ep_ret']` in `collect_trajectories` (`main.py`
   lines ~40-120) are index-paired correctly per episode — a silent misalignment (e.g. a
   goal from one env's episode getting paired with another env's return) would inject
   systematically wrong labels that compound as `goal_buffer`/`success_buffer` (maxlen
   5000) accumulate over a long run, which would fit the "gets worse over time, faster on
   longer/deeper-buffer runs" pattern observed.
3. **Empirically check the `goal_buffer` category mix** over training (not just assumed
   from `goals_distribution`/epsilon-floor argument): sample `goal_buffer`'s actual
   contents at several points in a run and tabulate what fraction are impossible vs.
   grasp/plants/herbivores/carnivores. If impossibles become underrepresented as the goal
   sampler steers away from near-zero-LP categories, that's a real class-imbalance
   explanation for BCE miscalibration, separate from the reward-signal question above.
4. **Verify the `set_weights`/`update_buffer` positional-zip correctness** (flagged above)
   with an explicit one-off check — e.g. print parameter names+shapes from both
   `_sr_parameters_filter` and `_sr_delayed_parameters_filter` iterators side by side and
   confirm they correspond 1:1 before trusting the delayed-copy mechanism is doing what's
   intended.
5. **Instrument SR logit magnitude directly**: the existing `diag/sr_*` metrics in
   `main.py` (~line 367-374) only log the post-sigmoid mean over 8 fixed goals per
   category. Consider also logging the raw pre-sigmoid logit (mean/std) over training —
   the observation that `estimated_sr_impossibles`/`_grow_herbivores`/`_grow_carnivores`
   move in lockstep suggests the SR head may be collapsing toward one dominant output
   (e.g. a saturating bias term) rather than genuinely confusing distinct goal categories;
   raw logit magnitude over time would make this visible directly.
6. **Confirm `goal_sampler.step` cadence**: `recompute_freq=32` is compared against
   `self.step`, incremented once per `goal_sampler.update()` call — confirm this actually
   fires once per real SAC/SR gradient-update cycle (both configs have `nb_updates: 1`, so
   likely 1:1, but wasn't explicitly verified) so that `delay_depth * recompute_freq`
   corresponds to the intended number of real gradient steps of delay.

## Live run — do not disturb without checking in first

Run `9bc9ddc8a79d4568999c5560594fc154` (container `magellan-magellan-1`, via
`docker-compose.local.yml`, output `outputs/magellan_paper_delay99`, MLflow run name
`delay_depth_99`) is **still training** toward 500k episodes on the corrected
paper-fidelity config. As of the last check (ep 263000/500000): grasp/grow_plants solved,
VRAM stable ~12GB/24GB, no errors. This run is valuable evidence for whichever root cause
is eventually confirmed — don't stop it; instead correlate any code fix's predictions
against its full trajectory once it's further along or complete.

- Check status: `docker compose -f docker-compose.local.yml ps`,
  `nvidia-smi --query-gpu=... --format=csv,noheader`
- Check metrics: `/check-metrics 9bc9ddc8a79d4568999c5560594fc154` (or the MLflow UI at
  `http://localhost:5000/#/experiments/2/runs/9bc9ddc8a79d4568999c5560594fc154/model-metrics`)
- Latest findings file:
  `.claude/findings/20260727_202303_9bc9ddc8a79d4568999c5560594fc154.json`

## Related memory (auto-memory system, not in this repo)

The Claude session that did this investigation maintains cross-session memory at
`~/.claude/projects/-home-jordy-projects-MAGELLAN/memory/`. Relevant entries:
- `project_sr_head_drift.md` — the primary tracker for this bug, updated live during the
  investigation described above.
- `project_delay_depth_vs_N.md` — the config-gap findings (batch size, delay_depth) that
  were ruled out as the cause.
- `project_grasp_regression.md` — the prior, resolved, unrelated bug (context only).
