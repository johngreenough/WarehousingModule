import csv


def write_plan(plan, output_path):
    fieldnames = [
        "step",
        "policy",
        "order_id",
        "position",
        "item_type",
        "tote",
        "order_bin",
        "switched_tote",
        "switched_bin",
        "incremental_time",
        "cumulative_time",
        "score_time_component",
        "score_optionality_component",
        "total_score",
        "remaining_total_units",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in plan:
            writer.writerow(row)


def write_sorter_input(plan, output_path, num_conveyors):
    shape_columns_by_id = {
        0: "cirle",
        1: "pentagon",
        2: "trapezoid",
        3: "triangle",
        4: "star",
        5: "moon",
        6: "heart",
        7: "cross",
    }

    conv_shape_counts = {}
    for row in plan:
        conv_num = (int(row["order_bin"]) - 1) % num_conveyors
        shape_id = int(row["item_type"])
        if shape_id not in shape_columns_by_id:
            continue
        key = (conv_num, shape_id)
        conv_shape_counts[key] = conv_shape_counts.get(key, 0) + 1

    all_shape_ids = list(shape_columns_by_id.keys())
    shape_columns = [shape_columns_by_id[sid] for sid in all_shape_ids]
    conv_nums = sorted({conv for conv, _ in conv_shape_counts.keys()})
    fieldnames = ["conv_num"] + shape_columns

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for conv in conv_nums:
            out = {"conv_num": conv}
            for sid in all_shape_ids:
                col = shape_columns_by_id[sid]
                out[col] = conv_shape_counts.get((conv, sid), 0)
            writer.writerow(out)

