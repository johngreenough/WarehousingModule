# Warehousing Module

This repository studies tote sequencing for sortation: choosing the **order that totes enter the facility** to minimize processing time while preserving future flexibility (optionality).

## Repository Layout

- `models/` - notebook models and comparison notebooks
- `inputs/` - canonical input CSVs and generated run folders
- `outputs/` - model result CSVs
- `MSE433_M3_data_generator.ipynb` - input data generation

## Data Model

Three CSV files define demand and tote sources:

- `order_itemtypes.csv`
- `order_quantities.csv`
- `orders_totes.csv`

Interpretation rule:

- row index = order
- column position within row = item slot
- same row + same position across all files = one mapped item record

For a row with:

- types: `3,4`
- quantities: `1,1`
- totes: `1,3`

The order needs one unit of type 3 from tote 1 and one unit of type 4 from tote 3.

## Input Generation

Use `MSE433_M3_data_generator.ipynb` to create test data.

It writes:

- canonical files to `inputs/`
- 500 sampled scenarios to `inputs/runs/run_0001` ... `inputs/runs/run_0500`

## Objective Used in Models

Current notebooks optimize a shared composite objective:

`objective_score = total_time - OPTIONALITY_LAMBDA * optionality_score`

Where:

- `total_time` includes placement time and switching penalties
- `optionality_score` rewards choices that keep more feasible future actions open
- lower `objective_score` is better

## Model Notebooks

### 1) Exact Optimization

- `models/exact_mip_model.ipynb`

Finds exact best tote sequence for the objective (MIP when available, exact DP fallback).

Outputs include per-run and aggregate summaries in `outputs/`, including objective components.

### 2) Heuristic Baselines

- `models/baseline_greedy.ipynb`
- `models/baseline_beam.ipynb`
- `models/baseline_random.ipynb`

All baselines use the same objective and produce per-run outputs and all-runs summaries for comparison.

### 3) Model Output Comparison

- `models/compare_model_outputs.ipynb`

Reads summary CSVs and creates cross-model comparison tables:

- `outputs/model_output_comparison.csv`
- `outputs/model_output_comparison_summary.csv`

### 4) ML Competition (Three ML Models)

- `models/ml_competition.ipynb`

Trains and evaluates three ML policies against exact-labeled behavior:

- linear logistic model
- 1-hidden-layer neural net
- tree-based boosted stumps

Produces:

- `outputs/ml_models_per_run.csv`
- `outputs/ml_models_summary.csv`

## Run Selection in Notebooks

Most model notebooks support:

- `RUN_ID = None` -> use canonical files in `inputs/`
- `RUN_ID = <int>` -> use one run folder `inputs/runs/run_XXXX/`
- `RUN_ID = "all"` -> process all run folders under `inputs/runs/`

## Sorter Schema

Sorter-format CSV columns:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

`conv_num` is constrained to `0..3` (4 conveyors).

## Recommended Workflow

1. Run `MSE433_M3_data_generator.ipynb` to refresh `inputs/` and `inputs/runs/`.
2. Run `models/exact_mip_model.ipynb` and baseline notebooks with `RUN_ID = "all"`.
3. Run `models/compare_model_outputs.ipynb` for aggregate model ranking.
4. Run `models/ml_competition.ipynb` to compare ML policies and gaps vs exact.
# Warehousing Module

This project optimizes the **order of totes entering the sortation facility**.

Models live in `models/`, input data in `inputs/`, and outputs in `outputs/`.

## Data Inputs

Core input files:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

How to read them together:

- each row = one order
- each position in a row = one item slot for that order
- the same row + same position across all 3 files maps:
  - item type
  - quantity
  - source tote

Example (same row position-aligned):

- item types: `3,4`
- quantities: `1,1`
- totes: `1,3`

Means the order needs one unit of type 3 from tote 1 and one unit of type 4 from tote 3.

## Data Generator

- `MSE433_M3_data_generator.ipynb`

Generates:

- canonical files in `inputs/`
- 500 sampled runs in `inputs/runs/run_0001` ... `run_0500`

## Model Notebooks

### Exact Model

- `models/exact_mip_model.ipynb`

Uses exact optimization (MIP with exact DP fallback) for tote-entry sequencing.

### Baselines

- `models/baseline_greedy.ipynb`
- `models/baseline_beam.ipynb`
- `models/baseline_random.ipynb`

### Output Comparison

- `models/compare_model_outputs.ipynb`

Compares per-run model outputs from the summary CSVs.

### ML Competition (3 ML Models)

- `models/ml_competition.ipynb`

Trains and compares three ML sequencing policies:

- linear logistic model
- neural net (1 hidden layer)
- tree-based boosted stumps

ML models are trained using exact-policy labels and evaluated with the same objective.

## Objective

All current models use the composite objective:

`objective_score = total_time - OPTIONALITY_LAMBDA * optionality_score`

Where:

- `total_time` includes placement time and switch penalties
- `optionality_score` rewards sequences that preserve future flexibility
- lower `objective_score` is better

## Running Across Inputs

In model notebooks:

- `RUN_ID = None` -> canonical `inputs/`
- `RUN_ID = <int>` -> one run folder `inputs/runs/run_XXXX/`
- `RUN_ID = "all"` -> all run folders under `inputs/runs/`

## Sorter Schema

Sorter input/output schema:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

`conv_num` is constrained to `0..3` (4 conveyors).
# Warehousing Module

Models are located in `models/`.

## Exact Optimization Model

- `models/exact_mip_model.ipynb`

Purpose:

- Solve the tote-entry sequence exactly for each generated input instance.
- Uses MIP (`pulp`) if available; otherwise exact DP fallback.

This notebook explicitly explains:

- objective
- decision variables
- constraints

Outputs:

- `outputs/exact_mip_tote_sequence.csv`
- `outputs/optimized_input_from_exact_mip_model.csv`
- `outputs/exact_mip_summary.csv`

Input selection in notebook:

- set `RUN_ID = None` to use canonical files in `inputs/`
- set `RUN_ID = 1..500` to use `inputs/runs/run_XXXX/`

## Baseline Models (separate notebooks)

- `models/baseline_greedy.ipynb`
- `models/baseline_beam.ipynb`
- `models/baseline_random.ipynb`

Outputs:

- `outputs/optimized_input_from_baseline_greedy.csv`
- `outputs/optimized_input_from_baseline_beam.csv`
- `outputs/optimized_input_from_baseline_random.csv`
- plus sequence and summary CSV files for each baseline

Input selection in each baseline notebook:

- set `RUN_ID = None` for canonical `inputs/`
- set `RUN_ID = 1..500` for `inputs/runs/run_XXXX/`

## Data Generation

- `MSE433_M3_data_generator.ipynb`

Now saves:

- canonical current data to `inputs/`
- 500 sampled runs to `inputs/runs/run_0001` ... `run_0500`

## Input Files

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

## Sorter Output Schema

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

with `conv_num` constrained to `0..3`.
# Warehousing Module

This project is streamlined to two model types for tote-entry sequencing.

## 1) Exact Optimization Model

- `exact_mip_model.ipynb`

What it does:

- Solves exact tote-entry order for data in `inputs/`.
- Uses a MIP formulation when `pulp` is available.
- Falls back to an exact DP solver if MIP package is not installed.

Inside this notebook, the following are explicitly documented:

- objective function
- decision variables
- constraints

Outputs:

- `outputs/exact_mip_tote_sequence.csv`
- `outputs/optimized_input_from_exact_mip_model.csv`
- `outputs/exact_mip_summary.csv`

## 2) Baseline Heuristic Models (separate notebooks)

- `baseline_greedy.ipynb`
- `baseline_beam.ipynb`
- `baseline_random.ipynb`

These are simpler, faster methods used as comparison baselines.

Outputs:

- `outputs/optimized_input_from_baseline_greedy.csv`
- `outputs/optimized_input_from_baseline_beam.csv`
- `outputs/optimized_input_from_baseline_random.csv`
- plus corresponding sequence and summary CSVs.

## Inputs

Use the generated CSVs in `inputs/`:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

## Sorter Output Schema

All model outputs follow:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

with `conv_num` constrained to `0..3`.
# Warehousing Module

Primary objective: optimize the **order of totes entering the sortation facility**.

The project uses three top-level notebooks:

- `optimization_model.ipynb`
- `simulation_model.ipynb`
- `ml_model.ipynb`

## Inputs

Source files are in `inputs/`:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

Mapping rule:

- row = order
- column position = item slot
- same position across files links item type, quantity, and tote

## Notebook Outputs

### `optimization_model.ipynb`

- `outputs/optimized_input_from_optimization_model.csv`
- `outputs/optimization_model_plan.csv`
- `outputs/optimization_model_tote_entry_sequence.csv`
- `outputs/optimization_model_summary.csv`
- `outputs/optimization_model_exact_validation.csv`
- `outputs/optimization_model_exact_tote_sequence.csv`

### `simulation_model.ipynb`

- `outputs/simulation_model_results.csv`
- `outputs/simulation_model_best.csv`
- `outputs/simulation_model_best_tote_entry_sequence.csv`
- `outputs/optimized_input_from_simulation_model.csv`
- `outputs/simulation_model_exact_validation.csv`
- `outputs/simulation_model_exact_tote_sequence.csv`

### `ml_model.ipynb`

- `outputs/optimized_input_from_ml_model.csv`
- `outputs/ml_model_selected_config.csv`
- `outputs/ml_model_tote_entry_sequence.csv`
- `outputs/ml_model_exact_validation.csv`
- `outputs/ml_model_exact_tote_sequence.csv`

## Exact Optimality + Validation

Each notebook now includes:

- quantity-conservation validation
- an exact dynamic-programming tote-sequence benchmark (time-only objective)
- optimality-gap reporting against the exact benchmark

## Sorter Schema

Generated sorter-input files use:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

`conv_num` is constrained to `0..3`.

See `PROJECT_CONTEXT.md` for full technical context.
# Warehousing Module

The primary thing being optimized is the **order of totes entering the sortation facility**.

This repo uses three top-level model notebooks (no model folders):

- `optimization_model.ipynb`
- `simulation_model.ipynb`
- `ml_model.ipynb`

## Inputs

All source data is in `inputs/`:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

Alignment rule:

- row = order
- position in row = item slot
- same position across all 3 files links item type, quantity, and tote

## Models

### `optimization_model.ipynb`

Directly optimizes tote-entry sequence using optionality-aware scoring.

Outputs:

- `outputs/optimized_input_from_optimization_model.csv`
- `outputs/optimization_model_plan.csv`
- `outputs/optimization_model_tote_entry_sequence.csv`
- `outputs/optimization_model_summary.csv`
- `outputs/optimization_model_exact_validation.csv`
- `outputs/optimization_model_exact_tote_sequence.csv`

### `simulation_model.ipynb`

Runs many stochastic scenarios and selects the best tote-entry sequence from simulations.

Outputs:

- `outputs/simulation_model_results.csv`
- `outputs/simulation_model_best.csv`
- `outputs/simulation_model_best_tote_entry_sequence.csv`
- `outputs/optimized_input_from_simulation_model.csv`
- `outputs/simulation_model_exact_validation.csv`
- `outputs/simulation_model_exact_tote_sequence.csv`

### `ml_model.ipynb`

Uses simulation results to train a lightweight ML predictor and chooses a tote-entry configuration.

Outputs:

- `outputs/optimized_input_from_ml_model.csv`
- `outputs/ml_model_selected_config.csv`
- `outputs/ml_model_tote_entry_sequence.csv`
- `outputs/ml_model_exact_validation.csv`
- `outputs/ml_model_exact_tote_sequence.csv`

## Exact Optimality Add-On

Each notebook now includes an additional **exact dynamic-programming tote-sequence benchmark** (time-only objective for the same generated data and timing assumptions).

This provides:

- exact best tote sequence under the benchmark objective
- optimality gap of the notebook-selected sequence vs exact benchmark
- quantity-conservation validation checks

## Sorter Input Schema

Sorter-input files use:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

`conv_num` is constrained to 4 conveyors (`0..3`).

## Full Context

See `PROJECT_CONTEXT.md` for the full technical walkthrough.
# Warehousing Module

This project uses **three top-level model notebooks** to determine the best input into the sorter machine:

- `optimization_model.ipynb`
- `simulation_model.ipynb`
- `ml_model.ipynb`

For full technical context of all components, see `PROJECT_CONTEXT.md`.

## Data Inputs

Canonical inputs are in `inputs/`:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

Interpretation:

- each row = one order
- each position in the row = one item slot
- matching position across the 3 files gives:
  - item type
  - quantity
  - source tote

## Model Notebooks

### `optimization_model.ipynb`

Direct optimization model using optionality-aware sequencing.

Outputs:

- `outputs/optimized_input_from_optimization_model.csv`
- `outputs/optimization_model_plan.csv`
- `outputs/optimization_model_summary.csv`

### `simulation_model.ipynb`

Stochastic simulation over many sampled timing assumptions and policies to identify the best realized sorter input.

Outputs:

- `outputs/simulation_model_results.csv`
- `outputs/simulation_model_best.csv`
- `outputs/optimized_input_from_simulation_model.csv`

### `ml_model.ipynb`

Machine learning model (linear regression on simulation data) to predict best policy/parameter configuration, then generate sorter input from that predicted best setup.

Outputs:

- `outputs/optimized_input_from_ml_model.csv`
- `outputs/ml_model_selected_config.csv`

## Sorter Input Schema

Generated sorter input files use:

`conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`

`conv_num` is constrained to 4 conveyors (`0..3`).
# Warehousing Module

## Data Explanation

The order data is split across three CSV files (in `inputs/`) that must be read together:

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

### How row/position matching works

- Each row is one order (row 1 = order 1, row 2 = order 2, etc.).
- Inside a row, each position corresponds to one item entry for that order.
- The same row and same position must be matched across all three files.

For order `i`, position `j`:

- `order_itemtypes[i][j]` = item type
- `order_quantities[i][j]` = quantity of that item type
- `orders_totes[i][j]` = tote containing that item

If a row has trailing commas or blank values, those are empty/missing positions (no extra item in that slot).

### Example

If row 1 is:

- item types: `3,4`
- quantities: `1,1`
- tote numbers: `1,3`

Then order 1 has:

- item type 3 with quantity 1 in tote 1
- item type 4 with quantity 1 in tote 3

In short: **same row + same position across the three files = one item record for that order**.

## Example Files

### `MSE433_M3_Example input.csv`

This file is a per-order demand table.

- `conv_num` is the order/conveyor number.
- Each remaining column is a shape type (`circle`, `pentagon`, `trapezoid`, `triangle`, `star`, `moon`, `heart`, `cross`).
- The value in each shape column is how many of that shape are required for that `conv_num`.

Example: if a row has `triangle = 3` and `cross = 2`, that order needs 3 triangles and 2 crosses.

### `MSE433_M3_Example output.csv`

This file is an event/schedule-style output with one picked/processed item per row.

- `conv_num` identifies which order/conveyor the event belongs to.
- `shape` is a numeric shape ID.
- `time` is when that item event occurred.

In this example format, shape IDs follow the same shape order as the input columns:

- `0 = circle`
- `1 = pentagon`
- `2 = trapezoid`
- `3 = triangle`
- `4 = star`
- `5 = moon`
- `6 = heart`
- `7 = cross`

## Optionality-Based Optimizer

The project now includes `optimize_optionality.py`, which builds a pick sequence that balances:

- **Time minimization** (less tote switching + bin switching + placement time)
- **Future flexibility (optionality)** (keep future choices open)

### Optimization idea

At each step, the optimizer chooses the next unit to pick by maximizing:

`total_score = (-time_weight * incremental_time) + (optionality_weight * future_optionality)`

Where:

- `incremental_time` uses this timing model for each picked unit:
  - `place_time` (constant time to place/sort one item)
  - `+ tote_switch_time` if the source tote changed from the previous step
  - `+ bin_switch_time` if the destination order bin changed from the previous step
- `future_optionality` rewards states with:
  - many remaining actions,
  - many active orders,
  - many active totes,
  - balanced remaining workload across totes,
  - and penalizes actions that close out an order/tote too early (locking you in).

The optimizer supports two policies:

- `greedy`: chooses the best immediate action.
- `beam`: does limited lookahead (beam search) before choosing the next action.

### Run

From the project root:

`python3 optimize_optionality.py`

This reads (by default):

- `inputs/order_itemtypes.csv`
- `inputs/order_quantities.csv`
- `inputs/orders_totes.csv`

And writes:

- `outputs/optimized_sorter_input.csv` (sorter-ready input format)
- `outputs/optimized_pick_plan.csv` (detailed step-by-step plan)

### Tune behavior

You can tune objective trade-offs:

- `--time-weight` higher => prioritize faster immediate time.
- `--optionality-weight` higher => prioritize keeping future options open.
- `--tote-switch-time` sets the time cost when switching source totes.
- `--bin-switch-time` sets the time cost when switching destination order bins.
- `--place-time` sets per-unit place/sort time.
- `--start-tote` sets the initial source tote.
- `--start-order-bin` sets the initial destination order bin.
- `--num-conveyors` sets how many conveyors exist; output `conv_num` is constrained to `0..num_conveyors-1` (default `0..3`).
- `--policy` chooses `greedy`, `beam`, or `random` (use `both` for all).
- `--beam-width` / `--beam-depth` control lookahead breadth/depth for beam search.

Example:

`python3 optimize_optionality.py --time-weight 1.0 --optionality-weight 1.5 --tote-switch-time 4.0 --bin-switch-time 0.75 --place-time 1.75 --start-tote 1 --start-order-bin 1 --num-conveyors 4`

What this command means:

- `--time-weight 1.0`: baseline importance for saving time.
- `--optionality-weight 1.5`: optionality is weighted 1.5x, so the solver prefers choices that keep more future options open.
- `--tote-switch-time 4.0`: if the next item comes from a different tote group, add 4.0 seconds.
- `--bin-switch-time 0.75`: if the next item goes to a different order bin, add 0.75 seconds.
- `--place-time 1.75`: each item placement adds 1.75 seconds.
- `--start-tote 1`: first step assumes source tote context starts at tote 1.
- `--start-order-bin 1`: first step assumes destination bin context starts at order bin 1.

### Run simulations and compare policies

You can run Monte Carlo comparisons under timing uncertainty:

`python3 optimize_optionality.py --simulate-runs 100 --policy both`

This writes:

- `outputs/simulation_runs.csv`: one row per run per policy.
- `outputs/simulation_summary.csv`: mean/P90 metrics by policy.
- `outputs/pareto_front.csv`: non-dominated policies (best trade-offs).

`--policy both` runs `greedy`, `beam`, and `random` baseline for comparison.

### Sensitivity analysis

Run a deterministic parameter sweep:

`python3 optimize_optionality.py --sensitivity-runs --policy both`

This writes:

- `outputs/sensitivity_results.csv`

Default grid values:

- `--sens-tote-switch-grid 3.0,4.0,5.0`
- `--sens-bin-switch-grid 0.5,0.75,1.0`
- `--sens-place-time-grid 1.5,1.75,2.0`

### Validation (model-vs-model)

Run repeated stochastic validation and compare win rates:

`python3 optimize_optionality.py --validate-runs 120 --policy both`

This writes:

- `outputs/validation_runs.csv`
- `outputs/validation_summary.csv`

`time_win_rate` is the share of runs where that policy had the lowest total time.

### What Pareto output means

`outputs/pareto_front.csv` keeps only solutions where no other solution is strictly better in all objectives.

In this project the objectives are:

- lower `mean_total_time`
- lower `mean_tote_switches`
- lower `mean_bin_switches`
- higher `mean_optionality`

So a Pareto solution is a valid "best trade-off" point, even if it is not the single fastest.

## Project folders

- `models/` contains model entrypoints (e.g., `models/run_optimizer.py`)
- `models/optimizer/` contains modular optimizer code (`cli`, `policy`, `analysis`, `io`, `data`)
- `inputs/` contains source CSV inputs
- `outputs/` contains generated plans, simulations, sensitivity, and validation results

### Input interpretation (exactly what gets optimized)

The script reads the three CSV files row-by-row and position-by-position.

For order `i` (row `i`) and item slot `j` (column position `j`):

- `order_itemtypes[i][j]` -> item type ID
- `order_quantities[i][j]` -> how many units of that item to pick
- `orders_totes[i][j]` -> tote number for that item

Each `(order i, slot j)` becomes a task with `quantity` units.  
If quantity is `3`, that task appears as 3 pick actions over time.

Example (single row):

- item types: `3,4`
- quantities: `1,2`
- totes: `1,3`

This creates:

- pick 1 unit of type 3 from tote 1
- pick 2 units of type 4 from tote 3

So total units to schedule from this row = `1 + 2 = 3` picks.
