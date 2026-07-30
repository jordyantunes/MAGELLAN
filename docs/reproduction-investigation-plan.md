# Why we don't reproduce the paper: investigation plan

Reference run: `1488b7198b4c46c7b022628278fa34e2` (magellan, `local_gpu_config_magellan_4090_paper`,
stopped at ~ep 320k/500k). Checkpoints: `outputs/magellan/2026-07-28-20-41-58/` (53 dirs, 15 GB,
latest `317195`). MLflow container is up on `localhost:5000`.

Symptom: `test/grasp` = 1.0, `test/grow_plants` ≈ 0.9, `test/grow_herbivores` = 0.0,
`test/grow_carnivores` = 0.0 at 320k episodes. Paper (Fig. 5) reaches >90% in **all four**
categories within 500k, with herbivores crossing 0.1 at 60–80k.

---

## 1. Evidence already gathered (Phase 1 of systematic debugging)

### 1.1 The curriculum is starving the unsolved categories — measured, not inferred

`goal_buffer.pkl` / `success_buffer.pkl` at ep 317195 hold the **last 5000 training episodes**
with their outcomes. Categorised by parsing the goal prompt against LittleZoo's category lists:

| sampled category | episodes / 5000 | share | successes | uniform-sampling share |
|---|---|---|---|---|
| grasp (possible) | 2750 | **55.0%** | 2750 | ~16% |
| impossible (all kinds) | 2099 | 42.0% | 0 | ~80% |
| grow_plants | 133 | 2.7% | 133 | 3.2% |
| **grow_herbivores** | **13** | **0.26%** | 0 | 0.64% |
| **grow_carnivores** | **5** | **0.10%** | 0 | 0.128% |

Two things at once:
- 55% of all practice goes to a category with **SR = 1.000** — i.e. zero true learning progress.
- herbivores/carnivores get **less practice than uniform random sampling would give them**.

At that rate the agent has seen on the order of a few hundred herbivore episodes in 300k. A 7-action
sequence cannot be learned from that. **The curriculum is not merely unhelpful, it is anti-helpful.**

### 1.2 The LP signal carries no per-category information

`test/estimated_lp_*` (post-threshold LP on the 64-goal-per-category probe set) over training:

```
ep         grasp  grow_plants  grow_herb  grow_carn  impossibles
50012     0.0111       0.0000     0.0000     0.0000       0.0000
75001     0.0537       0.0388     0.0368     0.0366       0.0350   <-- all five equal
150002    0.0125       0.0406     0.0320     0.0337       0.0284   <-- all five equal
200006    0.0256       0.0233     0.0203     0.0216       0.0191   <-- all five equal
300006    0.0075       0.0012     0.0000     0.0000       0.0000
mean      0.0056       0.0060     0.0041     0.0042       0.0038
frac>0      0.34         0.28       0.20       0.20         0.22
```

LP pulses **globally, in lockstep, including on impossible goals**, and is zero ~70–80% of the time.
This is the "SR head lockstep" already logged in `.claude/findings/` and
`docs/lp-collapse-investigation.md`, now quantified in the form that matters: LP is a *global noise
signal*, not a per-goal competence-change signal. Consequences for `sample()`:

- LP all-zero → `sum_lp == 0` → **uniform fallback** over 25k goals (herbivores 0.64%).
- LP nonzero but flat → `p = lp/sum(lp)` ≈ uniform over goals, weighted by goal **counts**
  (4000 grasp vs 160 herbivore vs 32 carnivore) → still ~uniform.
- LP nonzero only on residual grasp noise (the ep-300k row) → **all** LP mass on the mastered
  category. This is what produced the 55% grasp share.

MAGELLAN has degenerated to the "Uniform" baseline or worse — and §A.4 of the paper states outright
that uniform sampling cannot reach the hard categories inside 500k episodes. **Our result is the
paper's own Uniform curve, not a MAGELLAN failure per se.**

### 1.3 Two concrete code/config deviations that can produce exactly this

**(a) LP window is 25× too short — `delay_depth`.**

Paper §C.2: *"The window for LP computation is defined by |B| × update frequency"*, Table 6: |B| = 100,
update frequency = 32. The sampler is updated once per policy-update cycle (`update_freq` = 64
episodes), so the paper's LP window is

```
100 snapshots x 32 sampler updates x 64 episodes = 204,800 episodes
```

Ours (`goal_sampler.py:250`, `delay_depth` omitted → default 3 → `_buff_size` = 4):

```
  4 snapshots x 32 sampler updates x 64 episodes =   8,192 episodes
```

With an 8k-episode window, the delayed SR estimator is nearly identical to the current one, so real
LP on a slowly-improving category is small — and then `lp[lp < 0.01] = 0.0`
(`goal_sampler.py:302`) **erases it entirely**, leaving only whatever short-term noise happens to
exceed 0.01. That is precisely the observed pattern: flat, pulsing, threshold-clipped LP.
With a 205k-episode window, `sr_delayed` for grasp at ep 300k is the ep-100k agent (already
mastered → LP ≈ 0) while a category being learned across that span shows a large, smooth LP.

The config comment argues `_buff_size = int(N/recompute_freq + 1) = 4` from the released code and
calls `delay_depth: 99` "an apparently incorrect reading of Table 6". Given §C.2's explicit
window formula and the LP data above, **that judgement now looks backwards** and is hypothesis H1.
Prior finding (memory): `delay_depth=99` was measured VRAM-safe at ~12 GB/24 GB.

**(b) Target entropy makes α decay monotonically to zero — `updater.py:176`.**

```python
self.target_entropy = lambda n: torch.log(torch.FloatTensor(1.0 / n)).unsqueeze(-1)   # = -log(n)
```

The α loss (`updater.py:332`) is `-α · (log π + T)`, whose gradient w.r.t. `log α` is `α·(H − T)`,
so equilibrium is at `H = T`. Here `T = -log(n) ≈ -2.1` for n≈8 candidate actions, i.e. a **negative
entropy target, which is unreachable** — `H ≥ 0 > T` always, so α decreases at every single update
regardless of the policy. Observed: `train/alpha` = 9.7e-7, `train/entropy` = 0.074 (peaked 1.4 at
ep 67k, then monotone decline). Paper Table 5 says target entropy = **−0.125**; with that value the
decay is ~17× slower and the policy is held near H ≈ 0.125 instead of collapsing. This is upstream
code (commit `20f2729`, initial import), not one of our edits — but it is a documented
code-vs-paper discrepancy, and a near-deterministic policy cannot discover a 10-action carnivore
sequence even if the curriculum hands it the goal. Hypothesis H2.

### 1.4 H1 is already partially falsified by a run we forgot we had

MLflow contains a run named **`delay_depth_99`** (`9bc9ddc8a79d4568999c5560594fc154`, reached
ep 271k) — i.e. the paper's |B| = 100 window, the exact fix H1 proposes:

```
                                  34k     68k    102k    136k    170k    204k    238k
eval/grow_plants                0.016   1.000   1.000   1.000   1.000   1.000   1.000
eval/grow_herbivores            0.000   0.016   0.000   0.000   0.000   0.000   0.000
eval/grow_carnivores            0.000   0.000   0.000   0.000   0.000   0.000   0.000
test/estimated_lp_grasp         0.062   0.079   0.094   0.030   0.025   0.022   0.174
test/estimated_lp_grow_herbiv.  0.090   0.079   0.166   0.020   0.052   0.027   0.138
test/estimated_lp_impossibles   0.092   0.085   0.174   0.021   0.059   0.025   0.173
train/entropy                   1.508   1.285   1.317   1.062   0.878   0.398   0.354
```

The long window does what it should to LP's *magnitude* (0.02–0.17 vs 0.00–0.05) and it keeps
entropy far healthier (0.35 vs 0.07 at a comparable episode). But LP remains **completely
non-discriminative** — `impossibles` has the *highest* LP at 102k and 238k, tracking grasp and
herbivores to within a few percent — and herbivores/carnivores are still 0.000 at 271k.

**Conclusion: fixing the LP window is necessary but not sufficient. The window was not the
binding constraint — the SR head's output is.**

### 1.5 The SR head appears not to be learning at all

At ep 300k the head predicts `estimated_sr_grasp` = **0.0735** for a category whose true SR is
**1.000** and which constitutes **55% of its own training data, all with label 1**
(§1.1). It predicts 0.041–0.073 for *every* category, impossible ones included. Meanwhile the
empirical success rate in `success_buffer` is 2883/5000 = **58%**, so the head is not even fitting
the base rate — a head that had learned nothing but the prior would sit near 0.58, not 0.05.

A head that outputs a near-constant ~0.05 regardless of input yields
`LP = |sr − sr_delayed| ≈ |global drift|`, identical for every goal — which is *exactly* the
lockstep signature in §1.2 and §1.4, across every run, at every `delay_depth`.

**And we cannot currently see the SR loss at all: `train/sr_loss` does not exist.** `sr_update`
deliberately returns nothing (commit `fef10c2` — gathering its return payload over the shared
lamorel IPC channel corrupted the actor/critic path), so the BCE loss is computed on the LLM worker
and discarded. We have been flying blind on the component that the whole method depends on.

### 1.6 Deviations audited and cleared (do not spend time here)

| item | ours | paper | verdict |
|---|---|---|---|
| `goals_distribution` | `[20000,4000,800,160,32]` | §A.2 n, n/5, n/5², n/5³, n/5⁴ | ✅ exact |
| test protocol | 64 goals/category every 5000 ep | §4.2 identical | ✅ exact (`utils/tests.py:17-20`) |
| SAC hyperparameters | lr 1e-4, a_lr 1e-3, γ .99, batch 256, update_freq 64, n_step 3, buffer 500k, warmup 10 | Table 5 | ✅ exact |
| MAGELLAN hyperparameters | ε 1→0.2/320, D 5000, batch 256, recompute_freq 32 | Table 6 | ✅ exact except |B| (H1) |
| `n_llm_processes` | 1 | 2 (data parallel) | ✅ semantically neutral; 2 impossible on 1 GPU (verified) |
| 4-bit quantisation | off | on | ✅ ours is strictly higher precision |
| herbivore episode horizon | 10 steps (LittleZoo) | Table 3 says 11 | ⚠️ upstream env, affects paper too — note only |

---

## 2. Ranked hypotheses

Reordered after §1.4/§1.5. The SR estimator, not the LP window, is now the prime suspect.

| # | hypothesis | mechanism | discriminating prediction |
|---|---|---|---|
| **H3** ⬆ | **the SR head does not train** (broken `sr_update`: wrong param set, gradients not reaching the head/adapters, or an optimizer that never effectively steps) | head ≈ constant → `LP = \|sr − sr_delayed\|` is a global drift term identical for all goals → curriculum carries zero information at *any* window length | offline: the ep-317k head has **no better than chance discrimination** (AUC ≈ 0.5) between successful and failed goals in its own `goal_buffer`, and its mean prediction (~0.05) is nowhere near the 0.58 base rate |
| **H1** ⬇ | LP window too short (`delay_depth` 3 vs 99) | real slow LP < 0.01 threshold → zeroed | **partially falsified**: `delay_depth_99` raised LP magnitude but left it non-discriminative and herbivores at 0.000 (§1.4). Still a required fidelity fix, no longer a candidate root cause on its own |
| **H2** | unreachable target entropy → α→0, H→0 | near-deterministic policy cannot explore 7–15 action sequences | forcing herbivore goals at the ep-317k checkpoint yields no SR rise despite abundant practice. Note `delay_depth_99` held entropy at 0.35 and *still* got 0.000 → H2 alone is likely not sufficient either |
| **H4** | `lp < 0.01 → 0` threshold | erases the only real signal | offline sampler replay without the threshold materially changes category shares |
| **H5** | reward/credit-assignment bug on long episodes (n-step 3, max_steps 10/15) | herbivore returns never reach the buffer correctly | replay-buffer audit shows no non-zero-reward herbivore transitions even in successful episodes |

The failure is now best read as **two independent faults stacked**: a competence estimator that
emits no usable signal (H3) — which reduces MAGELLAN to the paper's Uniform baseline — and an
entropy target that cannot be met (H2), which caps how hard a goal the policy can crack even when
it is handed one. Neither alone explains all the evidence; H3 explains why *every* run and every
`delay_depth` looks the same.

---

## 3. Plan

### Phase A — offline, zero GPU-hours, from artefacts already on disk

**A1. Full-training sampled-goal distribution.** `replay_buffer.pkl` (163 MB, capacity 500k) holds
every transition of the run; its states carry the goal prompt. Categorise as in §1.1 to get the
sampled-category share **as a function of episode**, not just the last 5000. Deliverable: a
share-vs-episode plot per category. This tells us *when* the curriculum broke (my expectation:
around ep 60–80k, coinciding with grasp mastery and the entropy peak).

**A2. Success-conditional audit (tests H5).** In the same pass, count herbivore/carnivore episodes
with non-zero reward and confirm terminal-reward transitions are present and correctly discounted.
If herbivores were *never once* successful, H5 is dead and the problem is upstream of learning.

**A3. Sampler replay without the threshold (tests H4).** Re-run `MAGELLANGoalSampler.sample()`
offline against the recorded per-goal LP with and without `lp[lp<0.01]=0`, report the resulting
category shares. Cheap, purely arithmetic.

### Phase B — the decisive cheap experiments (minutes-to-hours of GPU, no retraining)

**B0. Is the SR head learning anything? (tests H3 — now the first thing to run.)**
Purely offline, CPU/1-GPU, no training. Load the ep-317195 `sr_adapters` + SR MLP and score its own
`goal_buffer` (5000 goals with ground-truth outcomes, 58% positive):

- **AUC / accuracy** of `sr` against `success`. Chance (≈0.5) ⇒ H3 confirmed, the estimator is dead.
- **mean prediction vs 0.58 base rate** — a working head should at minimum match the prior.
- **per-category mean and spread** — is there *any* separation between grasp (label 1) and
  impossibles (label 0)?
- repeat at ~6 checkpoints (5.7k → 317k) to see whether discrimination ever existed and decayed,
  or never appeared.

If it is at chance, stop and debug `sr_update` directly (param filter `_sr_trainable_params`,
whether `optimizer_sr`'s generator captured a non-empty param set, whether `loss.backward()`
gradients reach the `sr` head under `peft_adapter=sr_adapters`) with a standalone unit test:
overfit the head on 32 fixed goals and assert the BCE loss falls. That test does not need the
training loop at all and is the cheapest possible confirmation.

**B0b. Restore SR-loss observability (prerequisite for any future run).**
`train/sr_loss` must exist before we launch anything. Since the IPC return path is off-limits
(`fef10c2`), write it from the LLM worker to a side channel — a file under the run's output dir, or
a separate MLflow run tagged to the parent — and never through `agent.update`'s return value.

**B1. Reconstruct the paper's LP window offline (secondary now that §1.4 exists — tests H1).**
We have 53 checkpoints spanning 5.7k→317k episodes, and each `model.checkpoint` contains the
`sr_adapters` + SR-head weights (`_all_params_filter`, `updater.py:362`). So:

- fix a probe set of ~64 goals per category (5 categories) from the training goal space;
- for each checkpoint, load `sr_adapters` and compute per-goal SR;
- build `LP_window(k) = |SR(ep) − SR(ep − k·Δ)|` per goal, for k corresponding to windows of
  8k (our setting) and ~205k (paper's) episodes.

If the 205k-window LP separates categories — near 0 on grasp/impossibles, large on the frontier
category — while the 8k-window LP is flat noise, **H1 is confirmed with no training at all**, and
we simultaneously get the correct value of `delay_depth` to use. If both windows are flat, H1 is
dead and H3 is promoted (the head itself is degenerate).

Cost: ~53 × 320 forward passes on flan-t5-base. Well under an hour.

**B2. Forced-curriculum probe (tests H2, isolates curriculum from policy).**
Resume from `outputs/magellan/2026-07-28-20-41-58/317195` with `goal_sampler=random` and a goal
space restricted to possible grow_herbivore goals (a temporary `goals_distribution` /
`generate_goals` filter, ~20k episodes, `test_freq=2000`).

- herbivore SR rises → the policy *can* learn it; the sole blocker is goal starvation (H1) →
  exploration collapse is not binding, and fixing the curriculum is sufficient.
- herbivore SR stays 0 with thousands of on-goal episodes → H2 is binding and the target-entropy
  fix is required *in addition*.

This is the single most informative experiment in the plan; run it in parallel with B1.

**B3. Entropy sensitivity check (quantifies H2, optional if B2 is positive).**
Same checkpoint, evaluation only: sample herbivore rollouts at softmax temperatures 1.0 / 1.5 / 2.0
and record SR. Non-zero SR at raised temperature = the sub-policies exist but are unreachable under
the collapsed distribution.

### Phase C — fix, then one validation run per variable

Only after A+B. Order and single-variable discipline matter because a full run is ~24–36 h here.

1. **C1: `delay_depth: 99`** in `local_gpu_config_magellan_4090_paper.yaml` (plus a
   `weights_buffer` memory check — 100 × ~7.5 MB ≈ 750 MB host RAM, and re-verify the ~12 GB VRAM
   figure from the earlier measurement). Nothing else changes.
2. **C2: target entropy** → make it a config key defaulting to the paper's `-0.125`, i.e.
   `target_entropy: -0.125` rather than `log(1/n)`. Apply only if B2/B3 showed exploration is
   binding. Note in the commit that this diverges from the released upstream code and why.
3. Re-run 500k episodes, single seed, and gate on the paper's own milestones as go/no-go
   checkpoints rather than waiting for the end:

Gate on `eval/*` (the training goal space — the paper's Fig. 5 comparison), not `test/*`; see §4.

| milestone | paper | abort if |
|---|---|---|
| **SR head AUC on `goal_buffer` > 0.7** | — (new instrument) | not by 30k — nothing downstream can work without it |
| grasp > 0.9 | 20–30k | not by 50k |
| grow_plants > 0.9 | 40–60k | not by 100k |
| **herbivore sampling share > 5%** once plants are mastered | — (new instrument) | not by 100k |
| grow_herbivores > 0.1 | 60–80k | not by 130k |
| grow_carnivores > 0.1 | 150–200k | not by 260k |

The third row is the one we were missing: it fails ~200k episodes earlier than the SR milestone and
is the direct readout of whether the curriculum works.

### Phase D — instrumentation to keep (add before C, it is what made this run hard to diagnose)

1. `train/sampled_share_{category}` — rolling share of sampled goals per category. Non-negotiable;
   §1.1 required a checkpoint autopsy to obtain a number that should be a live metric.
2. `train/lp_mass_{category}` — share of total LP mass per category (LP × goal count), which is
   what actually drives `p`.
3. `train/target_entropy` and `train/entropy_gap` (H − T) — makes an unreachable target obvious
   immediately.
4. Re-enable the `diag/*` block: it is present in `magellan/main.py` (uncommitted) but **absent
   from this run's MLflow metrics**, so the raw pre-threshold LP was unavailable for 320k episodes.
   Verify it actually logs before starting C.

---

## 4. Audit: did *our* changes break it?

Full review of `main..experiments` (23 commits) plus uncommitted working-tree changes, for
`magellan/` only. **Verdict: no regression on our branch explains the failure.** Details:

| file | change | verdict |
|---|---|---|
| `environment.py` | goal generation rewritten: enumerate-then-`np.random.choice` → **reservoir sampling** | ✅ equivalent. `_reservoir_add` is a correct Algorithm R (`rng.integers(0, count+1)`, replace if `j < k`) ⇒ uniform k-subset, same as before. Impossibility predicate rewritten with sets — checked term by term, semantics preserved |
| `environment.py` | `np.random.seed(seed)` (global) → local `default_rng(seed)` | ✅ harmless: `main.py:167` still seeds the global RNG. Previously `generate_goals` *re*-seeded it on each call; the random stream differs, the statistics do not |
| `environment.py` | test path `filter_test=True` | ⚠️ **real bug**: `_sample` is called twice with independent draws — once to fill `goals`, once to rebuild the category lists — so the lists desync from the dict. `test_policy` intersects them, silently shrinking e.g. grow_herbivores from 160 to ~11 distinct goals. **Not reached in this run** (`adaptation_test: false` ⇒ `filter_test=False` ⇒ `goals = all_goals`), but it *is* live for any `adaptation_test: true` (Q4) run. Fix before touching Q4 |
| `goal_sampler.py` | `buff_size = int(N/recompute_freq + 1)` → `delay_depth + 1` | ✅ **numerically identical** for this config: `int(100/32 + 1) = 4` and `3 + 1 = 4`. The short LP window (H1) is **inherited from upstream**, not introduced by us |
| `models.py` | pad mask double loop → `output_tokens == self._pad_token` | ✅ equivalent (old code defaulted True, cleared where token ≠ pad) |
| `updater.py` | entropy log `masked_fill(~mask, 0.0)` | ✅ fixes the `0 · -inf = NaN` logging bug. Log-only, no gradient path |
| `updater.py` | `torch.cuda.empty_cache()` → gated on `_empty_cache_between_chunks` | ✅ perf only |
| `updater.py` | `sr_update` no longer returns a payload (`fef10c2`) | ✅ the intended fix — but it is *why* we have no `train/sr_loss` (see §1.5, B0b) |
| `updater.py` | `perf.time(...)` wrappers throughout | ✅ **verified**: the apparent indentation shift of `critic_optimizer.step()` / `policy_optimizer.step()` / `a_optimizer.step()` into the gradient-accumulation loops is an artifact of the `with` blocks. All three remain *outside* their loops (`updater.py:268`, `:356` vs loops at `:209`, `:292`). No change to accumulation semantics |
| `utils/tests.py` | `test_lp` `replace=False` → `True` | ✅ fixes the documented OOB sampling crash; adds a little variance to `estimated_*` means |
| `utils/generate_prompt.py` | `*args, **kwargs` added | ✅ signature only |
| `main.py` | TF32 enabled on matmul/cudnn | ✅ still far above the paper's 4-bit quantisation. There is a `4090og_no_tf32` ablation run (`d2796f7c`) if we ever want to confirm |
| `main.py` | MLflow logging, timestamped `output_dir`, git hash, `run_name` | ✅ additive |
| `main.py` | `test_lp(test_goals, goal_sampler, config_args)` → 2 args | ✅ the documented arg-count fix |
| `main.py` | `goal_sampler.update()` indentation | ✅ **verified** unchanged nesting (`:473` `with` at 12 spaces = main's `:370`): still once per `while` iteration, outside the `nb_updates` loop. ε-decay rate and LP snapshot cadence are therefore unchanged, so the §1.3 window arithmetic holds for both branches |
| `main.py` | `diag/*` block added (`d775e19`) | ⚠️ **real bug**: it is a silent no-op. `[g for g in ... if c in g]` tests lowercase `'grasp'`/`'grow_plants'` against prompts reading `Goal: Grasp carrot` / `Goal: Grow cow`, so every category list is empty and nothing is logged. This is why **no `diag/*` metric exists in any run** and why the pre-threshold LP has been invisible for 320k episodes. Must be fixed before Phase C (§Phase D.4) |

Two further inherited (not ours) issues surfaced by the audit:

- **`eval/*` and `test/*` are the reverse of what the names suggest.** `main.py:218` sets
  `eval_goals = train_goals`, so `eval/*` is the **training** goal space (the paper's Fig. 5, §4.2
  target) and `test/*` is the **held-out** set (§4.3). All prior findings — including §1 of this
  document as first written — compared `test/*` against Fig. 5. **This is on `main` too, so it is
  upstream, and the conclusion is unaffected**: `eval/grow_herbivores` and `eval/grow_carnivores`
  are 0.000 for the entire run. One detail does change — on the train space `grow_plants` reaches a
  clean 1.000 with no wobble, so the 0.906 "dip" recorded at ep 300k is a held-out generalization
  artifact, not a training collapse. Report `eval/*` from now on.
- **`updater.py:176` target entropy** (§1.3b) is from the initial upstream import (`20f2729`), not
  one of our edits.

## 5. What I am *not* going to do

- No config change before B1/B2 report. §1.3 gives two plausible root causes and the temptation is
  to flip both `delay_depth` and the entropy target and launch — that would cost 30 h and leave us
  unable to attribute the outcome.
- No re-derivation of the goal-space or SAC hyperparameters: audited exact (§1.4).
