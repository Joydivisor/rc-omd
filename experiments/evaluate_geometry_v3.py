"""Apply the frozen `geometry-v3-2026-08-19` selection and confirmation rules.

Two modes matching the two protocol phases:

* ``--mode discovery`` computes the bias-corrected frontier advantage on the
  30 discovery scenarios, ranks the closed candidate menu by absolute Spearman
  correlation, and emits the selected predictor. It never emits a confirmation
  verdict and never touches the confirmation subset.
* ``--mode confirmation`` applies the already-frozen predictor to the 20
  confirmation scenarios and emits CONFIRMED / NOT CONFIRMED.

The leave-one-out bias correction of E10 is applied here as a pre-registered
step, not post hoc: for every uniform grid point, that point is removed and its
own AUC interpolated from the remainder at its own distractor KL, where the
true value is exactly zero by construction.

Spearman correlation and its bootstrap interval are computed directly rather
than through scipy, so the repository gains no runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.evaluate_geometry_v2 import (
    AUC_KEY,
    CRITICAL_KEY,
    DISTRACTOR_KEY,
    match_frontier,
    t_ci_95,
)

PROTOCOL_ID = "geometry-v3-2026-08-19"


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata's 'average' method."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average ranks within tied groups.
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denominator == 0.0:
        return 0.0
    return float((rx * ry).sum() / denominator)


def bootstrap_rho_interval(
    x: np.ndarray, y: np.ndarray, resamples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        pick = rng.integers(0, n, size=n)
        draws[index] = spearman(x[pick], y[pick])
    return {
        "rho": spearman(x, y),
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "resamples": resamples,
    }


def scenario_metric(
    methods: dict[str, Any], candidate: str, grid_prefix: str, max_dropped: int
) -> dict[str, Any]:
    """Bias-corrected frontier advantage plus the allocation invariant."""

    names = sorted(name for name in methods if name.startswith(grid_prefix))
    grid = [
        {
            "name": name,
            AUC_KEY: np.asarray(methods[name][AUC_KEY], float),
            DISTRACTOR_KEY: np.asarray(methods[name][DISTRACTOR_KEY], float),
            CRITICAL_KEY: np.asarray(methods[name][CRITICAL_KEY], float),
        }
        for name in names
    ]
    entry = methods[candidate]
    candidate_auc = np.asarray(entry[AUC_KEY], float)
    candidate_distractor = np.asarray(entry[DISTRACTOR_KEY], float)
    candidate_critical = np.asarray(entry[CRITICAL_KEY], float)

    primary = match_frontier(
        grid, candidate_distractor, candidate_auc, DISTRACTOR_KEY, AUC_KEY
    )
    dropped = primary["dropped_seeds"]

    # Pre-registered leave-one-out bias.
    per_point: list[float] = []
    for held in names:
        rest = [point for point in grid if point["name"] != held]
        held_entry = methods[held]
        result = match_frontier(
            rest,
            np.asarray(held_entry[DISTRACTOR_KEY], float),
            np.asarray(held_entry[AUC_KEY], float),
            DISTRACTOR_KEY,
            AUC_KEY,
        )
        if result["differences"].size:
            per_point.append(float(result["differences"].mean()))
    bias = float(np.mean(per_point)) if per_point else float("nan")

    allocation = match_frontier(
        grid,
        candidate_critical,
        candidate_distractor,
        CRITICAL_KEY,
        DISTRACTOR_KEY,
        log_readout=True,
    )
    ratio_deviation = float("nan")
    if allocation["reference"].size:
        ratios = (allocation["differences"] + allocation["reference"]) / allocation[
            "reference"
        ]
        ratio_deviation = float(np.abs(ratios - 1.0).max())

    usable = dropped <= max_dropped
    interval = t_ci_95(primary["differences"]) if usable else None
    return {
        "delta_raw": interval["mean"] if interval else None,
        "delta_ci": interval,
        "loo_bias": bias,
        "delta_corrected": (interval["mean"] - bias) if interval else None,
        "dropped_seeds": dropped,
        "usable": usable,
        "allocation_max_abs_deviation_from_one": ratio_deviation,
    }


def metrics_for(config, summary, names) -> dict[str, Any]:
    rule = config["decision_rule"]
    return {
        name: scenario_metric(
            summary[name],
            str(rule["candidate_method"]),
            str(rule["grid_prefix"]),
            int(rule["max_dropped_seeds"]),
        )
        for name in names
    }


def run_discovery(config, summary) -> dict[str, Any]:
    rule = config["decision_rule"]
    discovery = list(config["split"]["discovery"])
    control_name = str(config["control_scenario"])

    metrics = metrics_for(config, summary, discovery)
    control = scenario_metric(
        summary[control_name], str(rule["candidate_method"]),
        str(rule["grid_prefix"]), int(rule["max_dropped_seeds"]),
    )
    invariant_holds = (
        control["allocation_max_abs_deviation_from_one"]
        <= float(rule["invariant_tolerance"])
    )

    usable = [name for name in discovery if metrics[name]["usable"]]
    response = np.array([metrics[name]["delta_corrected"] for name in usable])
    pooled_bias = float(np.mean([metrics[name]["loo_bias"] for name in usable]))

    predictors = config["predictors"]
    menu = list(next(iter(predictors.values())).keys())
    correlations = []
    for position, candidate in enumerate(menu):
        values = np.array([predictors[name][candidate] for name in usable])
        rho = spearman(values, response) if len(set(values.tolist())) > 1 else 0.0
        correlations.append(
            {"predictor": candidate, "rho": rho, "abs_rho": abs(rho),
             "menu_position": position}
        )
    ranked = sorted(correlations, key=lambda c: (-c["abs_rho"], c["menu_position"]))
    best = ranked[0]
    threshold = float(rule["selection_threshold_abs_rho"])

    if not invariant_holds:
        outcome = "HALT_INVARIANT_VIOLATED"
    elif not (pooled_bias > 0.0):
        outcome = "HALT_BIAS_CORRECTION_UNJUSTIFIED"
    elif best["abs_rho"] >= threshold:
        outcome = "SELECTED"
    else:
        outcome = "NO_PREDICTOR_FOUND"

    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "discovery",
        "scenarios": discovery,
        "usable_scenarios": usable,
        "excluded_for_coverage": [n for n in discovery if n not in usable],
        "metrics": metrics,
        "pooled_loo_bias": pooled_bias,
        "control": {"scenario": control_name, **control,
                    "invariant_holds": invariant_holds},
        "correlations": ranked,
        "selected": best if outcome == "SELECTED" else None,
        "selection_threshold_abs_rho": threshold,
        "refuted_predictor_selected": (
            outcome == "SELECTED"
            and best["predictor"] in ("pure_crit", "pure_crit_indicator")
        ),
        "outcome": outcome,
    }


def run_confirmation(config, summary, predictor: str) -> dict[str, Any]:
    rule = config["decision_rule"]
    confirmation = list(config["split"]["confirmation"])
    metrics = metrics_for(config, summary, confirmation)
    usable = [name for name in confirmation if metrics[name]["usable"]]

    response = np.array([metrics[name]["delta_corrected"] for name in usable])
    values = np.array([config["predictors"][name][predictor] for name in usable])
    interval = bootstrap_rho_interval(
        values, response, int(rule["bootstrap_resamples"]),
        int(rule["bootstrap_seed"]),
    )
    bound = float(rule["confirmation_rho_lower_bound"])
    confirmed = interval["ci_lower"] > bound

    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "confirmation",
        "predictor": predictor,
        "scenarios": confirmation,
        "usable_scenarios": usable,
        "metrics": metrics,
        "spearman": interval,
        "required_ci_lower_bound": bound,
        "outcome": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor", default=None,
                        help="required in confirmation mode; must be the frozen one")
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))

    if arguments.mode == "discovery":
        result = run_discovery(config, summary)
        print(f"outcome: {result['outcome']}  pooled bias "
              f"{result['pooled_loo_bias']:+.5f}")
        for entry in result["correlations"]:
            print(f"  {entry['predictor']:<30}rho {entry['rho']:+.4f}")
    else:
        if not arguments.predictor:
            raise SystemExit("--predictor is required in confirmation mode")
        result = run_confirmation(config, summary, arguments.predictor)
        interval = result["spearman"]
        print(f"outcome: {result['outcome']}")
        print(f"  {result['predictor']}: rho {interval['rho']:+.4f} "
              f"[{interval['ci_lower']:+.4f}, {interval['ci_upper']:+.4f}] "
              f"vs required lower bound {result['required_ci_lower_bound']}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
