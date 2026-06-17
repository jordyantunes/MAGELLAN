# MAGELLAN Paper Outline & Search Index

**Full title:** MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces
**Source:** https://arxiv.org/html/2502.07709v3
**Local copy:** `docs/magellan_paper.html`
**Extracted chart benchmarks (D.2/D.3/D.4):** `docs/paper_benchmarks.md`
**Chart images:** `docs/paper_figures/`

---

## How to use this outline

Each section heading links to an HTML anchor in the local file. Open with:
```
firefox docs/magellan_paper.html#<anchor>
```
or search with:
```
grep -n "<anchor>" docs/magellan_paper.html
```

---

## Structure

### Abstract
*[no anchor — top of document]*
One-paragraph summary of MAGELLAN's contribution: LLM-based ALP estimation that eliminates the need for expert goal groupings.

---

### H2: 1 Introduction `[#S1]`
- Motivates the curriculum learning problem in large goal spaces
- Introduces the metacognitive framing: the LLM "knows what it doesn't know"
- **Figure 1** `[#S1.F1]` — Overview diagram: how MAGELLAN uses past/current competence estimates to compute ALP per goal

---

### H2: 2 Related Work `[#S2]`

#### H3: 2.1 Goal selection in autotelic agents `[#S2.SS1]`
Prior work on intrinsic motivation and curriculum learning (IMGEP, playground, etc.)

#### H3: 2.2 Computing LP over goals `[#S2.SS2]`
Survey of ALP estimation methods — evaluation-based vs. sliding-window online methods; sets up the gap MAGELLAN fills
- **Table 1** `[#S3.T1]` — Comparison of ALP estimation methods across: computational efficiency, competence transfer, expert knowledge required

#### H3: 2.3 Autonomous LLM agents `[#S2.SS3]`
Recent LLM agent work; why LLMs are a natural fit for open-ended text-based environments

---

### H2: 3 Methods `[#S3]`

#### H3: 3.1 Problem statement `[#S3.SS1]`
Formal definition: goal space G, policy π, competence function C, learning progress ALP(g) = |C_θt(g) - C_θt-N(g)|

#### H3: 3.2 Metacognitive generalization of LP in LLM agents `[#S3.SS2]`
Core MAGELLAN algorithm:
- Competence estimator: LLM final-layer repr → MLP head → success probability (binary cross-entropy)
- Historical buffer: deque of past estimator weights at intervals of N steps
- ALP computed by querying current vs. delayed estimator on any goal

#### H3: 3.3 Classic ALP baselines `[#S3.SS3]`
- **Online-ALP**: sliding success-rate buffer per goal
- **EK-Online-ALP**: same but uses expert-defined goal-type buckets
- **Eval-ALP / EK-Eval-ALP**: periodic dedicated evaluation episodes

#### H3: 3.4 The Little-Zoo environment as a testbed `[#S3.SS4]`
- Fully text-based RL environment
- ~20 million goal combinations; 4 hidden goal families: grasp, grow_plants, grow_herbivores, grow_carnivores
- 80% of goals are impossible (realistic difficulty variation)
- **Figure 2** `[#S3.F2]` — A) Little-Zoo tech tree; B) goal space structure showing impossible vs. possible goals

---

### H2: 4 Experiments `[#S4]`

#### H3: 4.1 Q1 — How well does MAGELLAN estimate competence? `[#S4.SS1]`
- Competence estimation error vs. Eval-ALP (zero additional evaluation episodes needed)
- Scales to larger goal spaces
- **Figure 3** `[#S4.F3]` — Competence estimation error and cost vs. goal space size (scaling plot)
- **Figure 4** `[#S4.F4]` — Competence estimation on OpenR1-Math-220k (Algebra, Geometry, Number Theory)

#### H3: 4.2 Q2 — Training an LLM agent with MAGELLAN `[#S4.SS2]`
- MAGELLAN achieves 90%+ SR across all categories; Online-ALP fails without expert groupings
- **Figure 5** `[#S4.F5]` — SR evolution per goal category over 40k episodes (mean ± std across 8 seeds)
- **Table 2** `[#S4.T2]` — Predicted vs. observed competence on held-out test set per method

#### H3: 4.3 Q3 — MAGELLAN's generalization abilities `[#S4.SS3]`
- ~11% average error on unseen test goals
- LLM embeddings cluster goals semantically; impossible goals form a separate cluster
- **Figure 6** `[#S4.F6]` (a/b) — Generalization results

#### H3: 4.4 Q4 — MAGELLAN's adaptation to evolving goal spaces `[#S4.SS4]`
- Goal space replacement mid-training; MAGELLAN adapts faster when prior LP exists
- **Figure 7** `[#S4.F7]` (a/b) — Adaptation curves

---

### H2: 5 Conclusion `[#S5]`
Summary of contributions; future work directions (broader environments, other LLM sizes)

---

### H2: Impact Statement
Broader societal impact discussion

---

### H2: Acknowledgment

---

### H2: References

---

## Appendices

### H2: Appendix A — Little-Zoo Environment `[#A1]`

#### H3: A.1 Environment mechanics `[#A1.SS1]`
Step-by-step rules: item interactions, grow mechanics, hunger cycles

#### H3: A.2 Goal space generation `[#A1.SS2]`
How the full goal space is constructed from entity combinations

#### H3: A.3 Example of impossible goals `[#A1.SS3]`
Concrete examples of goals that cannot be achieved and why

#### H3: A.4 Goal repartition `[#A1.SS4]`
- **Table 3** `[#A1.T3]` — Optimal trajectories per category
- **Figure 8** `[#A1.F8]` (a/b) — Goal distribution: full space vs. experimental subset

---

### H2: Appendix B — Comparison of LP Methods `[#A2]`
Extended version of Table 1
- **Table 4** `[#A2.T4]` — Full prior-work comparison across efficiency, transfer assumptions, expert knowledge

---

### H2: Appendix C — Implementation Details `[#A3]`

#### H3: C.1 LLM-based RL agent `[#A3.SS1]`
SAC hyperparameters, LoRA adapter setup, Lamorel integration
- **Table 5** `[#A3.T5]` — SAC hyperparameters

#### H3: C.2 MAGELLAN `[#A3.SS2]`
MLP head architecture, training procedure, buffer management
- **Table 6** `[#A3.T6]` — MAGELLAN hyperparameters

#### H3: C.3 Baselines `[#A3.SS3]`
Online-ALP and Eval-ALP implementation specifics
- **Table 7** `[#A3.T7]` — Online methods hyperparameters
- **Table 8** `[#A3.T8]` — Eval methods hyperparameters

#### H3: C.4 Compute budget `[#A3.SS4]`
GPU hours, hardware used

---

### H2: Appendix D — Additional Results `[#A4]`

#### H3: D.1 Ablations on the MAGELLAN architecture `[#A4.SS1]`
Four architecture variants (A/B/C/D): different MLP depths, adapter placements
- **Figure 9** `[#A4.F9]` (a/b/c/d) — Ablation training curves
- **Figure 10** `[#A4.F10]` — Training curves for all four architectures
- **Figure 11** `[#A4.F11]` (a/b/c/d) — Per-architecture breakdown

#### H3: D.2 Q1. Competence estimation properties `[#A4.SS2]`

##### D.2.1 Per-goal competence estimation `[#A4.SS2.SSS1]`
- **Figure 12** `[#A4.F12]` — Competence estimation per method/category at 25k goals
- **Figure 13** `[#A4.F13]` — Same at 50k goals
- **Figure 14** `[#A4.F14]` — Same at 100k goals

##### D.2.2 Competence estimation on BabyAI-text goal-space `[#A4.SS2.SSS2]`
- **Figure 15** `[#A4.F15]` — MAGELLAN vs. Online-ALP on BabyAI-Text (5 difficulty levels)

##### D.2.3 Impact of the LLM used in MAGELLAN `[#A4.SS2.SSS3]`
- **Figure 16** `[#A4.F16]` — LLM choice effect on OpenR1-Math-220k (all models similar)
- **Figure 17** `[#A4.F17]` — LLM choice effect on BabyAI-Text

#### H3: D.3 Q2. Training an LLM-based RL agent with MAGELLAN `[#A4.SS3]`

##### D.3.1 Per-goal Success Rate `[#A4.SS3.SSS1]`
- **Figure 18** `[#A4.F18]` — Average SR per method/category (8 seeds)

##### D.3.2 Goal sampling strategies `[#A4.SS3.SSS2]`
- **Figure 19** `[#A4.F19]` (a/b/c) — Sampling distributions: MAGELLAN vs. EK-Online-ALP vs. Online-ALP
- **Figure 20** `[#A4.F20]` (a/b/c) — Goal selection heatmaps

#### H3: D.4 Q3. MAGELLAN's generalization abilities `[#A4.SS4]`

##### D.4.1 Per-goal success probability estimation on test set `[#A4.SS4.SSS1]`
- **Figure 21** `[#A4.F21]` (a–e) — Embedding visualizations at 5 training stages (nothing → grasp → plants → herbivores → carnivores mastered)

##### D.4.2 Evolution of the embeddings `[#A4.SS4.SSS2]`
- **Figure 22** `[#A4.F22]` (a/b) — Embedding evolution over training

##### D.4.3 Embedding of impossible goals `[#A4.SS4.SSS3]`
- **Figure 23** `[#A4.F23]` (a–j) — Impossible goal cluster analysis

#### H3: D.5 Q4. Leveraging generalization when facing new goals `[#A4.SS5]`

##### D.5.1 10 adaptation cases throughout training `[#A4.SS5.SSS1]`
Individual adaptation curves at 10 checkpoints across training

##### D.5.2 Global sample efficiency assessment `[#A4.SS5.SSS2]`
- **Figure 24** `[#A4.F24]` — Average sample efficiency after κ (test length) across 10 tests

---

## Quick-reference index

| Topic | Where to look |
|---|---|
| ALP formula / math | §3.1, §3.2 |
| MAGELLAN algorithm | §3.2, Appendix C.2, Table 6 |
| SAC hyperparameters | Appendix C.1, Table 5 |
| LoRA / adapter setup | Appendix C.1 |
| Little-Zoo rules | Appendix A.1, A.2 |
| Impossible goals | Appendix A.3, D.4.3 |
| Goal distribution used in experiments | Appendix A.4, Figure 8 |
| Competence estimation accuracy | §4.1, Appendix D.2, Figures 3, 12–15 |
| Training curves (main result) | §4.2, Figure 5, Appendix D.3, Figure 18 |
| Generalization to test goals | §4.3, Table 2, Appendix D.4 |
| Embedding visualizations | Appendix D.4.2, D.4.3, Figures 21–23 |
| Goal space adaptation | §4.4, Appendix D.5, Figure 24 |
| Baseline comparisons | §3.3, Appendix B, Table 4 |
| Ablation study | Appendix D.1, Figures 9–11 |
| BabyAI-Text cross-env results | Appendix D.2.2, Figure 15 |
| Math domain (OpenR1) results | §4.1 Figure 4, Appendix D.2.3, Figures 16–17 |
| Compute budget | Appendix C.4 |
