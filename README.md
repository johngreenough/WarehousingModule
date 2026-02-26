# Warehousing Module

## Data Explanation

The order data is split across three CSV files that must be read together:

- `order_itemtypes.csv`
- `order_quantities.csv`
- `orders_totes.csv`

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

This reads:

- `order_itemtypes.csv`
- `order_quantities.csv`
- `orders_totes.csv`

And writes:

- `optimized_sorter_input.csv` (sorter-ready input format)
- `optimized_pick_plan.csv` (detailed step-by-step plan)

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
- `--policy` chooses `greedy` or `beam`.
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

- `simulation_runs.csv`: one row per run per policy.
- `simulation_summary.csv`: mean/P90 metrics by policy.
- `pareto_front.csv`: non-dominated policies (best trade-offs).

### What Pareto output means

`pareto_front.csv` keeps only solutions where no other solution is strictly better in all objectives.

In this project the objectives are:

- lower `mean_total_time`
- lower `mean_tote_switches`
- lower `mean_bin_switches`
- higher `mean_optionality`

So a Pareto solution is a valid "best trade-off" point, even if it is not the single fastest.

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
