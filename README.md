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
