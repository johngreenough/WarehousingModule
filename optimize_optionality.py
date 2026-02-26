#!/usr/bin/env python3
"""
Compatibility entrypoint.
Core optimizer modules live under models/optimizer/.
"""

from models.optimizer import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hybrid warehouse picker optimizer:
- Minimize time (tote/bin switching + place time)
- Maximize optionality (future decision flexibility)
"""

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


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
        # Input data may be stored as "3.0"
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

            # Skip incomplete columns and zero/negative quantities.
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


def _incremental_time(
    current_tote,
    current_order_bin,
    next_tote,
    next_order_bin,
    tote_switch_time,
    bin_switch_time,
    place_time,
):
    switched_tote = 0
    switched_bin = 0

    if current_tote is not None and current_tote != next_tote:
        switched_tote = 1
    if current_order_bin is not None and current_order_bin != next_order_bin:
        switched_bin = 1

    step_time = place_time + (switched_tote * tote_switch_time) + (switched_bin * bin_switch_time)
    return step_time, switched_tote, switched_bin


def _entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if not probs:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    if len(probs) == 1:
        return 0.0
    return ent / math.log(len(probs))


def _future_optionality_score(tasks, candidate_idx):
    # Simulate taking one unit from candidate.
    remaining_snapshot = []
    for idx, t in enumerate(tasks):
        qty = t.remaining_qty - 1 if idx == candidate_idx else t.remaining_qty
        if qty > 0:
            remaining_snapshot.append((t.order_id, t.tote, qty))

    if not remaining_snapshot:
        return 0.0

    available_actions = sum(q for _, _, q in remaining_snapshot)
    distinct_orders = len({order for order, _, _ in remaining_snapshot})
    tote_counts = {}
    for _, tote, q in remaining_snapshot:
        tote_counts[tote] = tote_counts.get(tote, 0) + q
    distinct_totes = len(tote_counts)

    tote_entropy = _entropy(list(tote_counts.values()))

    # Penalize decisions that reduce branching by closing an order or tote.
    candidate = tasks[candidate_idx]
    closes_order = not any(
        (i != candidate_idx and t.order_id == candidate.order_id and t.remaining_qty > 0)
        or (i == candidate_idx and t.remaining_qty - 1 > 0)
        for i, t in enumerate(tasks)
    )
    closes_tote = not any(
        (i != candidate_idx and t.tote == candidate.tote and t.remaining_qty > 0)
        or (i == candidate_idx and t.remaining_qty - 1 > 0)
        for i, t in enumerate(tasks)
    )
    closure_penalty = (1 if closes_order else 0) + (1 if closes_tote else 0)

    return (
        0.20 * available_actions
        + 1.20 * distinct_orders
        + 1.40 * distinct_totes
        + 2.00 * tote_entropy
        - 1.50 * closure_penalty
    )


def _initial_remaining(tasks):
    return [t.remaining_qty for t in tasks]


def _active_candidates(remaining):
    return [i for i, qty in enumerate(remaining) if qty > 0]


def _future_optionality_score_from_state(tasks, remaining, candidate_idx):
    remaining_snapshot = []
    for idx, task in enumerate(tasks):
        qty = remaining[idx] - 1 if idx == candidate_idx else remaining[idx]
        if qty > 0:
            remaining_snapshot.append((task.order_id, task.tote, qty))

    if not remaining_snapshot:
        return 0.0

    available_actions = sum(q for _, _, q in remaining_snapshot)
    distinct_orders = len({order for order, _, _ in remaining_snapshot})
    tote_counts = {}
    for _, tote, qty in remaining_snapshot:
        tote_counts[tote] = tote_counts.get(tote, 0) + qty
    distinct_totes = len(tote_counts)
    tote_entropy = _entropy(list(tote_counts.values()))

    candidate = tasks[candidate_idx]
    closes_order = not any(
        (i != candidate_idx and t.order_id == candidate.order_id and remaining[i] > 0)
        or (i == candidate_idx and remaining[i] - 1 > 0)
        for i, t in enumerate(tasks)
    )
    closes_tote = not any(
        (i != candidate_idx and t.tote == candidate.tote and remaining[i] > 0)
        or (i == candidate_idx and remaining[i] - 1 > 0)
        for i, t in enumerate(tasks)
    )
    closure_penalty = (1 if closes_order else 0) + (1 if closes_tote else 0)

    return (
        0.20 * available_actions
        + 1.20 * distinct_orders
        + 1.40 * distinct_totes
        + 2.00 * tote_entropy
        - 1.50 * closure_penalty
    )


def _candidate_step(task, current_tote, current_order_bin, tote_switch_time, bin_switch_time, place_time):
    return _incremental_time(
        current_tote=current_tote,
        current_order_bin=current_order_bin,
        next_tote=task.tote,
        next_order_bin=task.order_id,
        tote_switch_time=tote_switch_time,
        bin_switch_time=bin_switch_time,
        place_time=place_time,
    )


def _pick_next_greedy(
    tasks,
    remaining,
    current_tote,
    current_order_bin,
    tote_switch_time,
    bin_switch_time,
    place_time,
    time_weight,
    optionality_weight,
):
    candidates = _active_candidates(remaining)
    best_idx = None
    best_tuple = None

    for idx in candidates:
        task = tasks[idx]
        dt, _, _ = _candidate_step(
            task, current_tote, current_order_bin, tote_switch_time, bin_switch_time, place_time
        )
        optionality = _future_optionality_score_from_state(tasks, remaining, idx)
        total_score = (-time_weight * dt) + (optionality_weight * optionality)
        ranking = (total_score, -dt, -task.task_id)
        if best_tuple is None or ranking > best_tuple:
            best_tuple = ranking
            best_idx = idx

    return best_idx


def _pick_next_random(remaining):
    candidates = _active_candidates(remaining)
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_next_beam(
    tasks,
    remaining,
    current_tote,
    current_order_bin,
    tote_switch_time,
    bin_switch_time,
    place_time,
    time_weight,
    optionality_weight,
    beam_width,
    beam_depth,
):
    beam = [
        {
            "remaining": remaining[:],
            "current_tote": current_tote,
            "current_order_bin": current_order_bin,
            "score": 0.0,
            "time": 0.0,
            "first_action": None,
        }
    ]

    for _ in range(max(1, beam_depth)):
        expanded = []
        for state in beam:
            candidates = _active_candidates(state["remaining"])
            for idx in candidates:
                task = tasks[idx]
                dt, _, _ = _candidate_step(
                    task,
                    state["current_tote"],
                    state["current_order_bin"],
                    tote_switch_time,
                    bin_switch_time,
                    place_time,
                )
                optionality = _future_optionality_score_from_state(tasks, state["remaining"], idx)
                step_score = (-time_weight * dt) + (optionality_weight * optionality)

                next_remaining = state["remaining"][:]
                next_remaining[idx] -= 1
                expanded.append(
                    {
                        "remaining": next_remaining,
                        "current_tote": task.tote,
                        "current_order_bin": task.order_id,
                        "score": state["score"] + step_score,
                        "time": state["time"] + dt,
                        "first_action": idx if state["first_action"] is None else state["first_action"],
                    }
                )

        if not expanded:
            break
        expanded.sort(key=lambda s: (s["score"], -s["time"]), reverse=True)
        beam = expanded[: max(1, beam_width)]

    if not beam:
        return None
    best_state = max(beam, key=lambda s: (s["score"], -s["time"]))
    return best_state["first_action"]


def optimize(
    tasks,
    start_tote,
    start_order_bin,
    tote_switch_time,
    bin_switch_time,
    place_time,
    time_weight,
    optionality_weight,
    policy,
    beam_width,
    beam_depth,
):
    remaining = _initial_remaining(tasks)
    current_tote = start_tote
    current_order_bin = start_order_bin
    cumulative_time = 0.0
    plan = []
    step = 1

    while any(qty > 0 for qty in remaining):
        if policy == "beam":
            chosen_idx = _pick_next_beam(
                tasks,
                remaining,
                current_tote,
                current_order_bin,
                tote_switch_time,
                bin_switch_time,
                place_time,
                time_weight,
                optionality_weight,
                beam_width,
                beam_depth,
            )
        elif policy == "random":
            chosen_idx = _pick_next_random(remaining)
        else:
            chosen_idx = _pick_next_greedy(
                tasks,
                remaining,
                current_tote,
                current_order_bin,
                tote_switch_time,
                bin_switch_time,
                place_time,
                time_weight,
                optionality_weight,
            )

        if chosen_idx is None:
            break

        chosen = tasks[chosen_idx]
        dt, switched_tote, switched_bin = _candidate_step(
            chosen, current_tote, current_order_bin, tote_switch_time, bin_switch_time, place_time
        )
        optionality = _future_optionality_score_from_state(tasks, remaining, chosen_idx)
        score_time_component = -time_weight * dt
        score_optionality_component = optionality_weight * optionality
        total_score = score_time_component + score_optionality_component

        cumulative_time += dt
        remaining[chosen_idx] -= 1
        current_tote = chosen.tote
        current_order_bin = chosen.order_id
        remaining_total_units = sum(remaining)

        plan.append(
            {
                "step": step,
                "policy": policy,
                "order_id": chosen.order_id,
                "position": chosen.position,
                "item_type": chosen.item_type,
                "tote": chosen.tote,
                "order_bin": chosen.order_id,
                "switched_tote": switched_tote,
                "switched_bin": switched_bin,
                "incremental_time": dt,
                "cumulative_time": cumulative_time,
                "score_time_component": score_time_component,
                "score_optionality_component": score_optionality_component,
                "total_score": total_score,
                "remaining_total_units": remaining_total_units,
            }
        )
        step += 1

    return plan


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
    # Sorter expects this exact schema/order (matching example input file).
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

    # Aggregate picked units into (conv_num, shape) counts.
    conv_shape_counts = {}
    for row in plan:
        # Map order bins into a fixed conveyor set (0..num_conveyors-1).
        conv_num = (int(row["order_bin"]) - 1) % num_conveyors
        shape_id = int(row["item_type"])
        if shape_id not in shape_columns_by_id:
            # Skip unknown shape IDs so sorter schema remains consistent.
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


def _plan_metrics(plan):
    if not plan:
        return {
            "steps": 0,
            "total_time": 0.0,
            "tote_switches": 0,
            "bin_switches": 0,
            "avg_optionality_component": 0.0,
        }

    return {
        "steps": len(plan),
        "total_time": plan[-1]["cumulative_time"],
        "tote_switches": sum(int(r["switched_tote"]) for r in plan),
        "bin_switches": sum(int(r["switched_bin"]) for r in plan),
        "avg_optionality_component": statistics.mean(float(r["score_optionality_component"]) for r in plan),
    }


def _is_dominated(point, points):
    # Minimize total_time and switches; maximize optionality.
    for other in points:
        if other is point:
            continue
        no_worse = (
            other["mean_total_time"] <= point["mean_total_time"]
            and other["mean_tote_switches"] <= point["mean_tote_switches"]
            and other["mean_bin_switches"] <= point["mean_bin_switches"]
            and other["mean_optionality"] >= point["mean_optionality"]
        )
        strictly_better = (
            other["mean_total_time"] < point["mean_total_time"]
            or other["mean_tote_switches"] < point["mean_tote_switches"]
            or other["mean_bin_switches"] < point["mean_bin_switches"]
            or other["mean_optionality"] > point["mean_optionality"]
        )
        if no_worse and strictly_better:
            return True
    return False


def run_simulations(tasks, args):
    policies = ["greedy", "beam", "random"] if args.policy == "both" else [args.policy]
    run_rows = []

    for run_id in range(1, args.simulate_runs + 1):
        tote_switch_time = max(
            0.0,
            random.gauss(
                args.tote_switch_time,
                max(0.0001, args.tote_switch_time * args.sim_time_noise),
            ),
        )
        bin_switch_time = max(
            0.0,
            random.gauss(
                args.bin_switch_time,
                max(0.0001, args.bin_switch_time * args.sim_time_noise),
            ),
        )
        place_time = max(
            0.0,
            random.gauss(
                args.place_time,
                max(0.0001, args.place_time * args.sim_time_noise),
            ),
        )

        for policy in policies:
            plan = optimize(
                tasks=tasks,
                start_tote=args.start_tote,
                start_order_bin=args.start_order_bin,
                tote_switch_time=tote_switch_time,
                bin_switch_time=bin_switch_time,
                place_time=place_time,
                time_weight=args.time_weight,
                optionality_weight=args.optionality_weight,
                policy=policy,
                beam_width=args.beam_width,
                beam_depth=args.beam_depth,
            )
            m = _plan_metrics(plan)
            run_rows.append(
                {
                    "run_id": run_id,
                    "policy": policy,
                    "tote_switch_time": tote_switch_time,
                    "bin_switch_time": bin_switch_time,
                    "place_time": place_time,
                    "total_time": m["total_time"],
                    "steps": m["steps"],
                    "tote_switches": m["tote_switches"],
                    "bin_switches": m["bin_switches"],
                    "avg_optionality_component": m["avg_optionality_component"],
                }
            )

    runs_path = Path(args.sim_output)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("w", newline="") as f:
        fieldnames = [
            "run_id",
            "policy",
            "tote_switch_time",
            "bin_switch_time",
            "place_time",
            "total_time",
            "steps",
            "tote_switches",
            "bin_switches",
            "avg_optionality_component",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    summary_by_policy = {}
    for row in run_rows:
        summary_by_policy.setdefault(row["policy"], []).append(row)

    summary_rows = []
    for policy, rows in summary_by_policy.items():
        total_times = sorted(float(r["total_time"]) for r in rows)
        p90_idx = max(0, min(len(total_times) - 1, int(math.ceil(0.9 * len(total_times))) - 1))
        summary_rows.append(
            {
                "policy": policy,
                "runs": len(rows),
                "mean_total_time": statistics.mean(float(r["total_time"]) for r in rows),
                "p90_total_time": total_times[p90_idx],
                "mean_tote_switches": statistics.mean(float(r["tote_switches"]) for r in rows),
                "mean_bin_switches": statistics.mean(float(r["bin_switches"]) for r in rows),
                "mean_optionality": statistics.mean(float(r["avg_optionality_component"]) for r in rows),
            }
        )

    summary_path = Path(args.sim_summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "policy",
            "runs",
            "mean_total_time",
            "p90_total_time",
            "mean_tote_switches",
            "mean_bin_switches",
            "mean_optionality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    pareto_rows = [row for row in summary_rows if not _is_dominated(row, summary_rows)]
    pareto_path = Path(args.pareto_output)
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    with pareto_path.open("w", newline="") as f:
        fieldnames = [
            "policy",
            "runs",
            "mean_total_time",
            "p90_total_time",
            "mean_tote_switches",
            "mean_bin_switches",
            "mean_optionality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pareto_rows:
            writer.writerow(row)

    return runs_path, summary_path, pareto_path, len(run_rows), len(pareto_rows)


def _parse_float_grid(text):
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def run_sensitivity_analysis(tasks, args):
    policies = ["greedy", "beam", "random"] if args.policy == "both" else [args.policy]
    tote_grid = _parse_float_grid(args.sens_tote_switch_grid)
    bin_grid = _parse_float_grid(args.sens_bin_switch_grid)
    place_grid = _parse_float_grid(args.sens_place_time_grid)

    rows = []
    case_id = 1
    for tote_switch_time in tote_grid:
        for bin_switch_time in bin_grid:
            for place_time in place_grid:
                for policy in policies:
                    plan = optimize(
                        tasks=tasks,
                        start_tote=args.start_tote,
                        start_order_bin=args.start_order_bin,
                        tote_switch_time=tote_switch_time,
                        bin_switch_time=bin_switch_time,
                        place_time=place_time,
                        time_weight=args.time_weight,
                        optionality_weight=args.optionality_weight,
                        policy=policy,
                        beam_width=args.beam_width,
                        beam_depth=args.beam_depth,
                    )
                    m = _plan_metrics(plan)
                    rows.append(
                        {
                            "case_id": case_id,
                            "policy": policy,
                            "tote_switch_time": tote_switch_time,
                            "bin_switch_time": bin_switch_time,
                            "place_time": place_time,
                            "total_time": m["total_time"],
                            "steps": m["steps"],
                            "tote_switches": m["tote_switches"],
                            "bin_switches": m["bin_switches"],
                            "avg_optionality_component": m["avg_optionality_component"],
                        }
                    )
                case_id += 1

    out_path = Path(args.sensitivity_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "policy",
            "tote_switch_time",
            "bin_switch_time",
            "place_time",
            "total_time",
            "steps",
            "tote_switches",
            "bin_switches",
            "avg_optionality_component",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path, len(rows)


def run_validation(tasks, args):
    policies = ["greedy", "beam", "random"] if args.policy == "both" else [args.policy]
    rows = []
    for run_id in range(1, args.validate_runs + 1):
        tote_switch_time = max(
            0.0,
            random.gauss(
                args.tote_switch_time,
                max(0.0001, args.tote_switch_time * args.sim_time_noise),
            ),
        )
        bin_switch_time = max(
            0.0,
            random.gauss(
                args.bin_switch_time,
                max(0.0001, args.bin_switch_time * args.sim_time_noise),
            ),
        )
        place_time = max(
            0.0,
            random.gauss(
                args.place_time,
                max(0.0001, args.place_time * args.sim_time_noise),
            ),
        )

        per_policy = []
        for policy in policies:
            plan = optimize(
                tasks=tasks,
                start_tote=args.start_tote,
                start_order_bin=args.start_order_bin,
                tote_switch_time=tote_switch_time,
                bin_switch_time=bin_switch_time,
                place_time=place_time,
                time_weight=args.time_weight,
                optionality_weight=args.optionality_weight,
                policy=policy,
                beam_width=args.beam_width,
                beam_depth=args.beam_depth,
            )
            m = _plan_metrics(plan)
            rec = {
                "run_id": run_id,
                "policy": policy,
                "total_time": m["total_time"],
                "tote_switches": m["tote_switches"],
                "bin_switches": m["bin_switches"],
                "avg_optionality_component": m["avg_optionality_component"],
            }
            per_policy.append(rec)
            rows.append(rec)

        if per_policy:
            best_time = min(r["total_time"] for r in per_policy)
            for r in per_policy:
                r["time_win"] = 1 if r["total_time"] == best_time else 0

    summary = []
    grouped = {}
    for r in rows:
        grouped.setdefault(r["policy"], []).append(r)

    for policy, data in grouped.items():
        summary.append(
            {
                "policy": policy,
                "runs": len(data),
                "mean_total_time": statistics.mean(d["total_time"] for d in data),
                "mean_tote_switches": statistics.mean(d["tote_switches"] for d in data),
                "mean_bin_switches": statistics.mean(d["bin_switches"] for d in data),
                "mean_optionality": statistics.mean(d["avg_optionality_component"] for d in data),
                "time_win_rate": statistics.mean(d.get("time_win", 0) for d in data),
            }
        )

    details_path = Path(args.validation_output)
    summary_path = Path(args.validation_summary_output)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with details_path.open("w", newline="") as f:
        fieldnames = [
            "run_id",
            "policy",
            "total_time",
            "tote_switches",
            "bin_switches",
            "avg_optionality_component",
            "time_win",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "policy",
            "runs",
            "mean_total_time",
            "mean_tote_switches",
            "mean_bin_switches",
            "mean_optionality",
            "time_win_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    return details_path, summary_path, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize pick order for time and optionality."
    )
    parser.add_argument("--itemtypes", default="inputs/order_itemtypes.csv")
    parser.add_argument("--quantities", default="inputs/order_quantities.csv")
    parser.add_argument("--totes", default="inputs/orders_totes.csv")
    parser.add_argument(
        "--output",
        default="outputs/optimized_sorter_input.csv",
        help="Sorter input CSV in the same format as MSE433_M3_Example input.csv",
    )
    parser.add_argument(
        "--plan-output",
        default="outputs/optimized_pick_plan.csv",
        help="Detailed unit-by-unit optimized pick plan (debug/analysis).",
    )
    parser.add_argument(
        "--num-conveyors",
        type=int,
        default=4,
        help="Number of conveyors in sorter input (conv_num range is 0..num_conveyors-1).",
    )
    parser.add_argument(
        "--policy",
        choices=["greedy", "beam", "random", "both"],
        default="greedy",
        help="Optimization policy. Use 'both' with --simulate-runs to compare policies.",
    )
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--beam-depth", type=int, default=8)
    parser.add_argument("--start-tote", type=int, default=None)
    parser.add_argument("--start-order-bin", type=int, default=None)
    parser.add_argument("--tote-switch-time", type=float, default=4.0)
    parser.add_argument("--bin-switch-time", type=float, default=0.75)
    parser.add_argument("--place-time", type=float, default=1.75)
    parser.add_argument(
        "--time-weight",
        type=float,
        default=1.0,
        help="Higher value prioritizes faster completion.",
    )
    parser.add_argument(
        "--optionality-weight",
        type=float,
        default=1.0,
        help="Higher value prioritizes future flexibility.",
    )
    parser.add_argument(
        "--simulate-runs",
        type=int,
        default=0,
        help="If > 0, runs Monte Carlo simulations under timing uncertainty.",
    )
    parser.add_argument(
        "--sim-time-noise",
        type=float,
        default=0.15,
        help="Relative stddev for sampled times in simulations (e.g., 0.15 = 15%).",
    )
    parser.add_argument("--sim-output", default="outputs/simulation_runs.csv")
    parser.add_argument("--sim-summary-output", default="outputs/simulation_summary.csv")
    parser.add_argument("--pareto-output", default="outputs/pareto_front.csv")
    parser.add_argument("--sensitivity-runs", action="store_true")
    parser.add_argument("--sens-tote-switch-grid", default="3.0,4.0,5.0")
    parser.add_argument("--sens-bin-switch-grid", default="0.5,0.75,1.0")
    parser.add_argument("--sens-place-time-grid", default="1.5,1.75,2.0")
    parser.add_argument("--sensitivity-output", default="outputs/sensitivity_results.csv")
    parser.add_argument("--validate-runs", type=int, default=0)
    parser.add_argument("--validation-output", default="outputs/validation_runs.csv")
    parser.add_argument(
        "--validation-summary-output",
        default="outputs/validation_summary.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    tasks = build_tasks(
        Path(args.itemtypes), Path(args.quantities), Path(args.totes)
    )
    if not tasks:
        raise SystemExit("No valid tasks found in input CSV files.")

    if args.num_conveyors <= 0:
        raise SystemExit("--num-conveyors must be >= 1.")
    if args.beam_width <= 0 or args.beam_depth <= 0:
        raise SystemExit("--beam-width and --beam-depth must be >= 1.")

    if args.sensitivity_runs:
        out_path, n_rows = run_sensitivity_analysis(tasks, args)
        print(f"Wrote {n_rows} sensitivity rows to {out_path}")
        return

    if args.validate_runs > 0:
        details_path, summary_path, n_rows = run_validation(tasks, args)
        print(f"Wrote {n_rows} validation rows to {details_path}")
        print(f"Wrote validation summary to {summary_path}")
        return

    if args.simulate_runs > 0:
        sim_policy = "both" if args.policy == "both" else args.policy
        args.policy = sim_policy
        runs_path, summary_path, pareto_path, n_rows, n_pareto = run_simulations(tasks, args)
        print(f"Wrote {n_rows} simulation rows to {runs_path}")
        print(f"Wrote simulation summary to {summary_path}")
        print(f"Wrote {n_pareto} Pareto-optimal rows to {pareto_path}")
        return

    selected_policy = "greedy" if args.policy == "both" else args.policy
    plan = optimize(
        tasks=tasks,
        start_tote=args.start_tote,
        start_order_bin=args.start_order_bin,
        tote_switch_time=args.tote_switch_time,
        bin_switch_time=args.bin_switch_time,
        place_time=args.place_time,
        time_weight=args.time_weight,
        optionality_weight=args.optionality_weight,
        policy=selected_policy,
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
    )

    output_path = Path(args.output)
    plan_output_path = Path(args.plan_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sorter_input(plan, output_path, args.num_conveyors)
    write_plan(plan, plan_output_path)

    total_time = plan[-1]["cumulative_time"] if plan else 0.0
    print(f"Wrote sorter input to {output_path}")
    print(f"Wrote {selected_policy} action plan ({len(plan)} rows) to {plan_output_path}")
    print(f"Estimated total time: {total_time:.3f}")


if __name__ == "__main__":
    main()
