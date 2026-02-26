import csv
import math
import random
import statistics
from pathlib import Path

from .policy import optimize


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


def _parse_float_grid(text):
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def run_simulations(tasks, args):
    policies = ["greedy", "beam", "random"] if args.policy == "both" else [args.policy]
    run_rows = []

    for run_id in range(1, args.simulate_runs + 1):
        tote_switch_time = max(0.0, random.gauss(args.tote_switch_time, max(0.0001, args.tote_switch_time * args.sim_time_noise)))
        bin_switch_time = max(0.0, random.gauss(args.bin_switch_time, max(0.0001, args.bin_switch_time * args.sim_time_noise)))
        place_time = max(0.0, random.gauss(args.place_time, max(0.0001, args.place_time * args.sim_time_noise)))

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
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "runs",
                "mean_total_time",
                "p90_total_time",
                "mean_tote_switches",
                "mean_bin_switches",
                "mean_optionality",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    pareto_rows = [row for row in summary_rows if not _is_dominated(row, summary_rows)]
    pareto_path = Path(args.pareto_output)
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    with pareto_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "runs",
                "mean_total_time",
                "p90_total_time",
                "mean_tote_switches",
                "mean_bin_switches",
                "mean_optionality",
            ],
        )
        writer.writeheader()
        for row in pareto_rows:
            writer.writerow(row)

    return runs_path, summary_path, pareto_path, len(run_rows), len(pareto_rows)


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
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path, len(rows)


def run_validation(tasks, args):
    policies = ["greedy", "beam", "random"] if args.policy == "both" else [args.policy]
    rows = []

    for run_id in range(1, args.validate_runs + 1):
        tote_switch_time = max(0.0, random.gauss(args.tote_switch_time, max(0.0001, args.tote_switch_time * args.sim_time_noise)))
        bin_switch_time = max(0.0, random.gauss(args.bin_switch_time, max(0.0001, args.bin_switch_time * args.sim_time_noise)))
        place_time = max(0.0, random.gauss(args.place_time, max(0.0001, args.place_time * args.sim_time_noise)))

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

    grouped = {}
    for r in rows:
        grouped.setdefault(r["policy"], []).append(r)

    summary = []
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "policy",
                "total_time",
                "tote_switches",
                "bin_switches",
                "avg_optionality_component",
                "time_win",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "runs",
                "mean_total_time",
                "mean_tote_switches",
                "mean_bin_switches",
                "mean_optionality",
                "time_win_rate",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    return details_path, summary_path, len(rows)

