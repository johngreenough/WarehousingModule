import math
import random


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

