import csv


class ItemTask:
    def __init__(self, task_id, order_id, position, item_type, tote, remaining_qty):
        self.task_id = task_id
        self.order_id = order_id  # 1-based row index
        self.position = position  # 1-based within-row position
        self.item_type = item_type
        self.tote = tote
        self.remaining_qty = remaining_qty


def _coerce_int(value):
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return int(float(stripped))
    except ValueError:
        return None


def _read_csv_rows(path):
    with path.open("r", newline="") as f:
        return list(csv.reader(f))


def build_tasks(itemtypes_path, quantities_path, totes_path):
    item_rows = _read_csv_rows(itemtypes_path)
    qty_rows = _read_csv_rows(quantities_path)
    tote_rows = _read_csv_rows(totes_path)

    n_orders = max(len(item_rows), len(qty_rows), len(tote_rows))
    tasks = []
    task_id = 1

    for order_idx in range(n_orders):
        item_row = item_rows[order_idx] if order_idx < len(item_rows) else []
        qty_row = qty_rows[order_idx] if order_idx < len(qty_rows) else []
        tote_row = tote_rows[order_idx] if order_idx < len(tote_rows) else []
        width = max(len(item_row), len(qty_row), len(tote_row))

        for pos_idx in range(width):
            item_type = _coerce_int(item_row[pos_idx]) if pos_idx < len(item_row) else None
            qty = _coerce_int(qty_row[pos_idx]) if pos_idx < len(qty_row) else None
            tote = _coerce_int(tote_row[pos_idx]) if pos_idx < len(tote_row) else None

            if item_type is None or qty is None or tote is None or qty <= 0:
                continue

            tasks.append(
                ItemTask(
                    task_id=task_id,
                    order_id=order_idx + 1,
                    position=pos_idx + 1,
                    item_type=item_type,
                    tote=tote,
                    remaining_qty=qty,
                )
            )
            task_id += 1

    return tasks

