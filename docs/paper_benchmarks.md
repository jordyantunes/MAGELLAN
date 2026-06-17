# Paper Benchmarks: D.2, D.3, D.4 Charts

Extracted from visual analysis of charts in `docs/magellan_paper.html`.
Source images saved in `docs/paper_figures/`.
All episode counts are approximate midpoints of the mean curve (8-seed runs).

---

## D.2 — Competence Estimation (Estimated Success Probability) [Figs 12–14]

**What is plotted:** Estimated success probability (y: 0–1.2) vs. episodes (x: 0–50k) for each method, compared against EK-Eval-ALP as ground truth. Three sub-figures differ by goal space size: 30k / 60k / 120k goals.

### Grasp
| Method       | ~Reaches 1.0 | Spread at convergence |
|---|---|---|
| MAGELLAN     | ~5–8k ep     | Tight (±0.05)         |
| EK-Online-ALP| ~5–8k ep     | Tight (±0.05)         |
| Online-ALP   | ~10k ep      | Wider (±0.15)         |
| EK-Eval-ALP  | ~5k ep       | Tight (reference)     |

### Grow Plants
| Method       | ~Reaches 1.0 | Spread at convergence |
|---|---|---|
| MAGELLAN     | ~10–15k ep   | Tight (±0.10)         |
| EK-Online-ALP| ~10–15k ep   | Tight (±0.10)         |
| Online-ALP   | ~20k ep      | Wide (±0.20–0.30)     |

### Grow Herbivores
| Method       | ~Value at 50k ep | Spread     |
|---|---|---|
| MAGELLAN     | ~0.85–1.0        | Moderate (±0.15) |
| EK-Online-ALP| ~0.85–1.0        | Moderate (±0.15) |
| Online-ALP   | ~0.75–0.90       | Wide (±0.25)     |

### Grow Carnivores
| Method       | ~Value at 50k ep | Spread     |
|---|---|---|
| MAGELLAN     | ~0.65–0.80       | Moderate (±0.20) |
| EK-Online-ALP| ~0.65–0.80       | Moderate (±0.20) |
| Online-ALP   | ~0.55–0.75       | Very wide (±0.35)|

**Scale effect (30k → 60k → 120k goals):** MAGELLAN spread stays roughly constant; Online-ALP spread grows substantially, especially on grow_herbivores and grow_carnivores.

---

## D.3 — Success Rate per Category [Fig 18]

**What is plotted:** Actual policy success rate (y: 0–1.0) vs. episodes (x: 0–500k) for MAGELLAN, Uniform, Online-ALP, EK-Online-ALP. This is the real policy SR, not competence estimation.

### Grasp
| Method        | ~Reaches 0.9 | Final SR  | Spread     |
|---|---|---|---|
| MAGELLAN      | ~20–30k ep   | ~1.0      | Tight      |
| EK-Online-ALP | ~20–30k ep   | ~1.0      | Tight      |
| Online-ALP    | ~30–40k ep   | ~1.0      | Moderate   |
| Uniform       | ~50k ep      | ~1.0      | Moderate   |

### Grow Plants
| Method        | ~Reaches 0.9 | Final SR  | Spread          |
|---|---|---|---|
| MAGELLAN      | ~80–100k ep  | ~1.0      | Tight           |
| EK-Online-ALP | ~80–100k ep  | ~1.0      | Tight           |
| Online-ALP    | Inconsistent | ~0.4–1.0  | Very wide (bimodal: some seeds succeed, others fail) |
| Uniform       | Never        | ~0.5–0.8  | Wide            |

### Grow Herbivores
| Method        | ~Starts rising (>0.1) | ~Reaches 0.5 | Final SR at 500k | Spread     |
|---|---|---|---|---|
| MAGELLAN      | ~60–70k ep            | ~150–175k ep | ~0.7–0.85        | Moderate   |
| EK-Online-ALP | ~60–70k ep            | ~150–175k ep | ~0.7–0.85        | Moderate   |
| Online-ALP    | Inconsistent          | Never stable | ~0.1–0.6         | Very wide  |
| Uniform       | Never                 | Never        | ~0.0             | Near zero  |

### Grow Carnivores
| Method        | ~Starts rising (>0.1) | ~Reaches 0.5 | Final SR at 500k | Spread     |
|---|---|---|---|---|
| MAGELLAN      | ~150–175k ep          | ~350–400k ep | ~0.5–0.65        | Large      |
| EK-Online-ALP | ~150–175k ep          | ~350–400k ep | ~0.4–0.65        | Very large |
| Online-ALP    | Never                 | Never        | ~0.0–0.1         | Near zero  |
| Uniform       | Never                 | Never        | ~0.0             | Near zero  |

**Key takeaway:** Only MAGELLAN and EK-Online-ALP reliably progress beyond Grasp+Plants. Grow Carnivores is still in progress at 500k episodes even for MAGELLAN — it is not "solved" within the paper's run budget.

---

## D.4 — Success Probability Estimation on Test Set [Figs 20a/b/c]

**What is plotted:** Observed (blue) vs. Estimated (orange) success probability on a held-out test goal set, across 500k episodes. Tests generalization of each method's competence model to unseen goals.

### MAGELLAN (sr_test_estimation_magellan.png)
| Category       | Observed at 500k | Estimated tracks? | Alignment quality |
|---|---|---|---|
| Grasp          | ~1.0             | Yes (~1.0)        | Excellent, tight spread |
| Grow Plants    | ~1.0             | Yes (~1.0)        | Good, minor overestimation during ramp |
| Grow Herbivores| ~0.6–0.8         | Yes (~0.6–0.8)    | Good, slight lag; moderate spread |
| Grow Carnivores| ~0.4–0.6         | Yes (~0.4–0.6)    | Reasonable; large spread on observed |

### EK-Online-ALP (sr_test_estimation_ek_online.png)
| Category       | Observed at 500k | Estimated tracks? | Alignment quality |
|---|---|---|---|
| Grasp          | ~1.0             | Yes (~1.0)        | Excellent |
| Grow Plants    | ~1.0             | Yes (~1.0)        | Good |
| Grow Herbivores| ~0.7–0.9         | Partial — overestimates during rise (~0.2 above observed) | Moderate |
| Grow Carnivores| ~0.4–0.6         | Noisy — diverges at times | Poor/variable |

### Online-ALP (sr_test_estimation_online.png)
| Category       | Observed at 500k | Estimated tracks? | Alignment quality |
|---|---|---|---|
| Grasp          | ~1.0             | Yes (~1.0)        | Good |
| Grow Plants    | ~0.8–1.0         | Yes (~0.8–1.0)    | Good |
| Grow Herbivores| ~0.5–0.7         | **No — flat ~0.0 throughout** | Fails completely |
| Grow Carnivores| ~0.0–0.1         | **No — flat ~0.0 throughout** | Fails completely |

**Key takeaway:** Online-ALP has zero competence generalization to test goals for grow_herbivores and grow_carnivores — it only tracks goals it has seen in training. MAGELLAN and EK-Online-ALP both generalize via their model (LLM embeddings vs. expert buckets respectively).

---

## Summary table for check-metrics comparisons

Use this table when classifying a run's metrics as `ABOVE_PAPER / ON_TRACK / BELOW_PAPER`:

| Metric                          | ABOVE_PAPER          | ON_TRACK range                         | BELOW_PAPER          |
|---|---|---|---|
| `test/grasp` reaches 0.9        | < 20k ep             | 20–50k ep                              | > 50k ep or never    |
| `test/grow_plants` reaches 0.9  | < 80k ep             | 80–120k ep                             | > 150k ep or never   |
| `test/grow_herbivores` first >0.1 | < 60k ep           | 60–80k ep                              | > 100k ep or never   |
| `test/grow_herbivores` at 500k  | > 0.85               | 0.60–0.85                              | < 0.60               |
| `test/grow_carnivores` first >0.1 | < 150k ep          | 150–200k ep                            | > 250k ep or never   |
| `test/grow_carnivores` at 500k  | > 0.65               | 0.40–0.65                              | < 0.40               |
| `diag/sr_impossibles` at 50k+   | < 0.05               | 0.05–0.15                              | > 0.20               |
| Estimated LP on test goals      | Tracks observed ±0.05| Tracks with ±0.05–0.15 lag/overshoot  | Flat ~0 on grow_*    |
| Competence est. error (Q1)      | < 0.05               | 0.05–0.15                              | > 0.20               |

**Note on run length:** Grow Carnivores is still converging at 500k episodes in the paper. For shorter runs (< 200k), not reaching grow_carnivores is `EXPECTED`, not `BELOW_PAPER`.
