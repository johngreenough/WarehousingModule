import argparse
import random
from pathlib import Path

from .analysis import run_sensitivity_analysis, run_simulations, run_validation
from .data import build_tasks
from .io_utils import write_plan, write_sorter_input
from .policy import optimize


def build_parser():
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    tasks = build_tasks(Path(args.itemtypes), Path(args.quantities), Path(args.totes))
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

