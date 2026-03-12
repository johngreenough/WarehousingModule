# Warehousing Module

This codebase optimizes the **order of totes entering the sortation facility**.

It is currently organized around one exact notebook, three baseline notebooks, one comparison notebook, and one data generator notebook.

## Current Codebase (What Exists Now)

- `MSE433_M3_data_generator.ipynb`
- `models/exact_mip_model.ipynb`
- `models/baseline_greedy.ipynb`
- `models/baseline_beam.ipynb`
- `models/baseline_random.ipynb`
- `models/compare_model_outputs.ipynb`
- `example/MSE433_M3_Example input.csv`
- `example/MSE433_M3_Example output.csv`

Directories:

- `inputs/runs/` - generated run folders (`run_0001`, etc.)
- `outputs/` - model outputs, summaries, and comparison artifacts

## Input Contract (Used by All Models)

Every run folder is expected to contain:

- `order_itemtypes.csv`
- `order_quantities.csv`
- `orders_totes.csv`

Interpretation:

- same row index across files = same order
- same position in row across files = same item slot
- quantity expands that slot into unit-level picks
- each unit references a source tote, so sequencing controls tote-entry order

Example:

- item types row: `3,4`
- quantities row: `1,1`
- totes row: `1,3`

means one unit of type 3 from tote 1 and one unit of type 4 from tote 3.

## Objective and Decision Logic

Models use a composite objective balancing time and future flexibility:

`objective_score = total_time - OPTIONALITY_LAMBDA * optionality_score`

Lower is better.

Where:

- `total_time` uses placement + tote-switch + bin-switch penalties
- `optionality_score` rewards preserving feasible future choices and avoiding early lock-in

## Notebook-by-Notebook Context

### `models/exact_mip_model.ipynb`

- Solves tote sequencing exactly (MIP).
- Produces run-level outputs and all-runs summary.
- Serves as the benchmark for gap/comparison analysis.

### `models/baseline_greedy.ipynb`

- Chooses next tote & item using immediate objective increment.
- Fast heuristic baseline.

### `models/baseline_beam.ipynb`

- Uses limited lookahead via beam search.
- Usually stronger than greedy with more compute.

### `models/baseline_random.ipynb`

- Samples random sequences and keeps best found.
- Provides a lower-quality baseline for dominance checks.

### `models/compare_model_outputs.ipynb`

- Aligns runs across exact + baselines.
- Writes model comparison tables.
- Computes paired gap statistics vs exact.
- Generates visualizations for distribution and gap analysis.

## Run Selection

Model notebooks support:

- `RUN_ID = None` -> canonical `inputs/` files (if present)
- `RUN_ID = <int>` -> one run folder in `inputs/runs/run_XXXX/`
- `RUN_ID = "all"` -> all run folders in `inputs/runs/`

Given this current codespace, `RUN_ID = "all"` is the expected default path.

## Validation and Comparative Analysis (Current)

Implemented in `models/compare_model_outputs.ipynb`:

- common-run alignment before comparing models
- mean objective + win counts
- paired tests using model gap vs exact
- plots for objective distributions and gaps

Practical checks you should keep using:

- schema consistency for sorter-output CSVs
- quantity conservation from inputs to generated plan/output
- reproducibility with fixed seeds and same run set
- gap-to-exact monitoring across regenerated data batches

## Visualizations

Comparison notebook generates model-level plots in `outputs/` (distribution, means, and gap views).

## Recommended Workflow (Current Codespace)

1. Run `MSE433_M3_data_generator.ipynb` to refresh `inputs/runs/`.
2. Run `models/exact_mip_model.ipynb` with `RUN_ID = "all"`.
3. Run baseline notebooks (`greedy`, `beam`, `random`) with `RUN_ID = "all"`.
4. Run `models/compare_model_outputs.ipynb` for summary tables, validation stats, and plots.
5. Review generated CSVs/plots in `outputs/`.
