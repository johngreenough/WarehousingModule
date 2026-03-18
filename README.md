# Warehousing Module

This codebase optimizes the **order in which totes are fed into a sortation facility** to minimize the sum of order completion times while preserving operational flexibility (optionality).

The sortation system consists of four conveyor belts connected in a square loop, with each side dedicated to fulfilling one customer order. Items from each tote are placed onto the belt and travel around the loop until picked off at their assigned order station. Each tote contains items belonging to multiple orders (typically 3–4), so the sequence in which totes are loaded — and items placed within each tote — directly affects how quickly orders complete and how smoothly the belt transitions between order stations.

---

## Files

```
MSE433_M3_data_generator.ipynb     — generates randomized input runs
models/exact_mip_model.ipynb       — exact MIP solver (Gurobi)
models/baseline_greedy.ipynb       — greedy heuristic
models/baseline_beam.ipynb         — beam search heuristic (beam width = 12)
models/baseline_random.ipynb       — random search (2000 tote trials, 200 item trials)
models/compare_model_outputs.ipynb — statistical comparison and plots
inputs/runs/                       — generated run folders (run_0001 … run_0050)
outputs/                           — model outputs, summaries, comparison artifacts
```

---

## Input Format

Every run folder contains three CSV files (no headers, ragged rows):

| File | Meaning |
|---|---|
| `order_itemtypes.csv` | Item type for each slot of each order |
| `order_quantities.csv` | Quantity for each slot of each order |
| `orders_totes.csv` | Tote assignment for each slot of each order |

Same row index = same order. Same column position = same item slot. Quantity expands a slot into that many unit-level picks, each referencing a source tote.

**Example:** item types row `3,4` / quantities row `1,1` / totes row `1,3` means one unit of type 3 from tote 1 and one unit of type 4 from tote 3.

The data generator caps items per tote at 5 (`MAX_ITEMS_PER_TOTE = 5`) to keep item-level MIP solves fast.

---

## Objective Function

All four models minimize the same composite objective:

```
objective = sum_order_completion_times - 0.35 × optionality_score
```

**Lower is better.**

### Sum of order completion times
The finish time of each order's last tote summed across all orders. An order is complete when the last tote containing any of its items has been fully processed. This penalizes sequences that leave orders partially fulfilled for long stretches, which causes downstream packing stations to wait.

### Optionality score
Rewards sequences that preserve flexibility for the operator — specifically, the ability to re-sequence remaining totes without incurring additional order-switch penalties. Built from three components:

- **Branch reward (×0.6, position-weighted):** rewards totes whose last order matches the first order of many other totes, placed early so that flexibility benefit is maximized
- **Rarity penalty (×−0.4):** discounts the position reward for totes whose first order is rare across the batch, since they are inherently hard to transition into smoothly
- **Edge bonus (×1.0):** flat reward for every consecutive tote pair where the last order of the departing tote matches the first order of the arriving tote — a seamless handoff with no order-switch penalty

### Cost parameters
| Parameter | Value | Meaning |
|---|---|---|
| `PLACE_TIME` | 1.75s | Time to place one item onto the belt |
| `TOTE_SWITCH_TIME` | 4.0s | Transition cost between any two consecutive totes |
| `ORDER_SWITCH_PENALTY` | 0.75s | Additional cost when consecutive items/totes belong to different orders |
| `OPTIONALITY_LAMBDA` | 0.35 | Weight on optionality term in objective |

---

## Models

### Exact MIP (`exact_mip_model.ipynb`)
Formulates tote sequencing as a Mixed Integer Program solved by Gurobi. Decides all tote-to-tote connections simultaneously, tracks exact finish times, and computes order completion times as hard constraints. Subtour elimination prevents the solver from forming disconnected loops. The same MIP structure is applied within each tote to sequence items. Time limits: 30s for the tote MIP, 5s per item MIP.

- **Strength:** guaranteed optimal (within time limit); most consistent results
- **Weakness:** requires Gurobi; not suitable for real-time decisions
- **Best used for:** offline wave/shift planning when the full tote set is known in advance

### Beam Search (`baseline_beam.ipynb`)
Builds the sequence one tote at a time, keeping the 12 best partial sequences alive at each step. All others are discarded. Same logic applied to item sequencing within each tote.

- **Strength:** structured look-ahead, escapes some greedy traps
- **Weakness:** inconsistent — high standard deviation on small instances; beam width may need tuning
- **Best used for:** medium to large batch sizes where random search degrades

### Greedy (`baseline_greedy.ipynb`)
At each step, commits immediately to whichever remaining tote produces the lowest incremental objective score. No look-ahead, no reconsideration.

- **Strength:** extremely fast; simplest to implement
- **Weakness:** worst-performing model; myopic choices can cause poor global outcomes
- **Best used for:** lower-bound baseline only

### Random (`baseline_random.ipynb`)
Generates 2,000 random tote orderings and 200 random item orderings per tote, keeps the best found.

- **Strength:** near-optimal on small batches (matched MIP exactly on 64% of runs); trivially parallelizable
- **Weakness:** degrades rapidly as batch size grows
- **Best used for:** small batches (≤ ~8 totes) where combinatorial coverage is achievable

---

## Statistical Results (50 runs)

| Model | Mean Objective | Std Dev | Win Count |
|---|---|---|---|
| Exact MIP | 229.08 | 127.16 | 50 / 50 |
| Random | 231.67 | 130.32 | 0 / 50 |
| Beam Search | 320.00 | 201.48 | 0 / 50 |
| Greedy | 335.73 | 194.97 | 0 / 50 |

All pairwise comparisons are statistically significant (paired t-test and Wilcoxon signed-rank, p < 0.05). Random performs within 1.1% of the MIP on average.

---

## Run Selection

All model notebooks support:

```python
RUN_ID = "all"   # all folders in inputs/runs/  (default)
RUN_ID = None    # canonical inputs/ files
RUN_ID = 3       # single run: inputs/runs/run_0003/
```

---

## Recommended Workflow

1. Run `MSE433_M3_data_generator.ipynb` — regenerates 50 runs in `inputs/runs/`
2. Run `models/exact_mip_model.ipynb` with `RUN_ID = "all"`
3. Run `models/baseline_greedy.ipynb`, `baseline_beam.ipynb`, `baseline_random.ipynb` with `RUN_ID = "all"`
4. Run `models/compare_model_outputs.ipynb` — produces summary tables, pairwise significance tests, and plots
5. Review outputs in `outputs/`

---

## Outputs

Each model writes per-run results to `outputs/<model>_runs/run_XXXX/` and an aggregate summary to `outputs/<model>_all_runs_summary.csv`.

The comparison notebook produces:

| File | Contents |
|---|---|
| `model_output_comparison.csv` | Per-run scores for all four models |
| `model_output_comparison_summary.csv` | Mean score and win count per model |
| `model_output_comparison_stats.csv` | Pairwise p-values (t-test + Wilcoxon) |
| Inline plots | Box plots, mean ± 95% CI, per-run line chart, difference histograms, p-value heatmap |
