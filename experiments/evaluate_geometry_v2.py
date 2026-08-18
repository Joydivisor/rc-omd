"""Apply the frozen `geometry-v2-2026-08-18` metrics and decision rules.

Implements `docs/GEOMETRY_V2_PROTOCOL.md` verbatim.

Primary metric, per scenario per seed: interpolate the swept uniform frontier's
success AUC at the candidate's cumulative distractor KL, linearly in log KL,
and report the paired difference. Brackets are chosen by KL *rank* rather than
grid adjacency, so the rule stays well defined where the per-seed ``KL(eta)``
curve is locally non-monotone and collapses to the adjacent bracket wherever it
is monotone.

Secondary metric: the same machinery with critical KL as the matching axis and
distractor KL as the read-out, interpolated geometrically. Geometric
interpolation is required rather than stylistic -- under complete aliasing
distractor KL equals critical KL identically, so matching on one and reading
the other must return the target, and only log-linear interpolation of the
read-out does that exactly.

Statistics follow `pareto-v1` and `geometry-v1`: paired per-seed differences
with two-sided 95% t-confidence bounds, t-criticals tabulated so the repository
gains no runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "geometry-v2-2026-08-18"

AUC_KEY = "success_auc_per_seed"
DISTRACTOR_KEY = "cumulative_distractor_kl_per_seed"
CRITICAL_KEY = "cumulative_critical_kl_per_seed"

# scipy.stats.t.ppf(0.975, df). The protocol fixes 20 seeds per cell and allows
# at most 2 out-of-bracket seeds to be dropped, so df is in {17, 18, 19}.
_T_CRITICAL_95 = {
    17: 2.1098155778333156,
    18: 2.1009220402410382,
    19: 2.0930240544083087,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def t_ci_95(values: np.ndarray) -> dict[str, float]:
    n = int(values.shape[0])
    df = n - 1
    if df not in _T_CRITICAL_95:
        raise ValueError(
            f"no tabulated two-sided 95% t-critical value for df={df}; "
            "geometry-v2 is frozen at 20 seeds/cell with at most 2 seeds "
            "droppable, so df must be in {17, 18, 19}"
        )
    mean = float(np.mean(values))
    half_width = _T_CRITICAL_95[df] * float(np.std(values, ddof=1)) / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": float(np.std(values, ddof=1)),
        "half_width": half_width,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def match_frontier(
    grid: list[dict[str, np.ndarray]],
    target_axis: np.ndarray,
    readout: np.ndarray,
    match_key: str,
    read_key: str,
    log_readout: bool = False,
) -> dict[str, Any]:
    """Interpolate the grid's ``read_key`` at the candidate's ``match_key``.

    Returns the paired per-seed differences, the interpolated reference values,
    and the indices of seeds that fell outside the grid's reachable range.
    """

    differences: list[float] = []
    reference: list[float] = []
    dropped: list[int] = []

    for seed_index in range(int(target_axis.shape[0])):
        target = float(target_axis[seed_index])
        low = high = None
        for point in grid:
            value = float(point[match_key][seed_index])
            if value <= target and (
                low is None or value > float(low[match_key][seed_index])
            ):
                low = point
            if value >= target and (
                high is None or value < float(high[match_key][seed_index])
            ):
                high = point
        if low is None or high is None:
            dropped.append(seed_index)
            continue

        match_low = float(low[match_key][seed_index])
        match_high = float(high[match_key][seed_index])
        read_low = float(low[read_key][seed_index])
        read_high = float(high[read_key][seed_index])

        if match_high <= match_low or match_low <= 0.0:
            # Degenerate bracket (duplicate or non-positive KL): the protocol
            # fixes the fallback to the lower point rather than extrapolating.
            value = read_low
        else:
            weight = (math.log(target) - math.log(match_low)) / (
                math.log(match_high) - math.log(match_low)
            )
            if log_readout:
                value = math.exp(
                    math.log(read_low)
                    + (math.log(read_high) - math.log(read_low)) * weight
                )
            else:
                value = read_low + (read_high - read_low) * weight

        reference.append(value)
        differences.append(float(readout[seed_index]) - value)

    return {
        "differences": np.asarray(differences, dtype=float),
        "reference": np.asarray(reference, dtype=float),
        "dropped_seeds": len(dropped),
        "dropped_index": dropped,
    }


def evaluate_scenario(
    scenario: str,
    methods: dict[str, Any],
    candidate: str,
    grid_prefix: str,
    max_dropped: int,
) -> dict[str, Any]:
    """Primary and secondary metrics with their verdicts, for one scenario."""

    grid_names = sorted(name for name in methods if name.startswith(grid_prefix))
    if not grid_names:
        raise ValueError(f"{scenario}: no grid methods with prefix {grid_prefix!r}")

    seeds = methods[candidate]["seeds"]
    grid: list[dict[str, Any]] = []
    for name in grid_names:
        entry = methods[name]
        if entry["seeds"] != seeds:
            raise ValueError(
                f"seed mismatch between {candidate} and {name} in {scenario}; "
                "pairing assumption violated"
            )
        grid.append(
            {
                "name": name,
                AUC_KEY: np.asarray(entry[AUC_KEY], dtype=float),
                DISTRACTOR_KEY: np.asarray(entry[DISTRACTOR_KEY], dtype=float),
                CRITICAL_KEY: np.asarray(entry[CRITICAL_KEY], dtype=float),
            }
        )

    entry = methods[candidate]
    candidate_auc = np.asarray(entry[AUC_KEY], dtype=float)
    candidate_distractor = np.asarray(entry[DISTRACTOR_KEY], dtype=float)
    candidate_critical = np.asarray(entry[CRITICAL_KEY], dtype=float)

    primary = match_frontier(
        grid, candidate_distractor, candidate_auc, DISTRACTOR_KEY, AUC_KEY
    )
    secondary = match_frontier(
        grid,
        candidate_critical,
        candidate_distractor,
        CRITICAL_KEY,
        DISTRACTOR_KEY,
        log_readout=True,
    )

    if primary["dropped_seeds"] > max_dropped:
        verdict = "inconclusive"
        primary_ci: dict[str, float] | None = None
    else:
        primary_ci = t_ci_95(primary["differences"])
        if primary_ci["ci_lower"] > 0.0:
            verdict = "positive"
        elif primary_ci["ci_upper"] < 0.0:
            verdict = "negative"
        else:
            verdict = "null"

    allocation: dict[str, Any] | None = None
    if secondary["dropped_seeds"] <= max_dropped and secondary["reference"].size:
        ratios = (secondary["differences"] + secondary["reference"]) / secondary[
            "reference"
        ]
        log_ci = t_ci_95(np.log(ratios))
        allocation = {
            "log_ratio": log_ci,
            "ratio_mean": math.exp(log_ci["mean"]),
            "ratio_ci": [math.exp(log_ci["ci_lower"]), math.exp(log_ci["ci_upper"])],
            "max_abs_deviation_from_one": float(np.abs(ratios - 1.0).max()),
        }

    # Descriptive only; excluded from every decision by the protocol.
    with np.errstate(divide="ignore", invalid="ignore"):
        descriptive = float(
            np.mean(np.log(candidate_distractor / candidate_critical))
        )

    return {
        "scenario": scenario,
        "candidate": candidate,
        "primary": {
            "delta_frontier": primary_ci,
            "dropped_seeds": primary["dropped_seeds"],
            "dropped_index": primary["dropped_index"],
            "verdict": verdict,
        },
        "secondary": {
            "allocation": allocation,
            "dropped_seeds": secondary["dropped_seeds"],
        },
        "descriptive_mean_log_dist_over_crit": descriptive,
    }


def check_invariant(
    scenario: str, result: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    """Control 1: under complete aliasing the allocation ratio must be exactly 1.

    This is a theorem about the parameterization, so a violation is an
    implementation defect and the protocol requires halting rather than
    reporting a result.
    """

    allocation = result["secondary"]["allocation"]
    if allocation is None:
        return {"scenario": scenario, "holds": False, "reason": "allocation unavailable"}
    worst = float(allocation["max_abs_deviation_from_one"])
    return {
        "scenario": scenario,
        "max_abs_deviation_from_one": worst,
        "tolerance": tolerance,
        "holds": worst <= tolerance,
    }


def decide(config: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    rule = config["decision_rule"]
    invariant_scenario = str(rule["invariant_scenario"])
    discriminator = str(rule["discriminating_scenario"])
    no_spurious = list(rule["no_spurious_pass_scenarios"])

    invariant = check_invariant(
        invariant_scenario, results[invariant_scenario],
        float(rule["invariant_tolerance"]),
    )

    verdicts = {name: result["primary"]["verdict"] for name, result in results.items()}

    def satisfies(name: str, predicted: str) -> bool:
        actual = verdicts[name]
        if predicted == "null_or_negative":
            return actual in ("null", "negative")
        return actual == predicted

    h_pure = all(
        satisfies(name, predicted)
        for name, predicted in rule["predictions_h_pure"].items()
    )
    h_hetero = all(
        satisfies(name, predicted)
        for name, predicted in rule["predictions_h_hetero"].items()
    )

    # Control 2: the zero margin must not fire on pure_crit = 0 scenarios.
    control_two = {
        name: verdicts[name] != "positive"
        for name in no_spurious
        if name != discriminator
    }
    control_two_holds = all(control_two.values())

    if not invariant["holds"]:
        outcome = "HALT_INVARIANT_VIOLATED"
    elif "inconclusive" in verdicts.values():
        outcome = "INCONCLUSIVE"
    elif h_pure and not h_hetero:
        outcome = "GO_H_PURE_CONFIRMED"
    elif h_hetero and not h_pure:
        outcome = "REVISE_H_HETERO_CONFIRMED"
    else:
        outcome = "NO-GO_LAW_REFUTED"

    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "test",
        "frozen_decision_rule": rule,
        "scenario_results": results,
        "verdicts": verdicts,
        "control_1_invariant": invariant,
        "control_2_no_spurious_pass": {
            "per_scenario": control_two,
            "holds": control_two_holds,
        },
        "hypothesis_h_pure_satisfied": h_pure,
        "hypothesis_h_hetero_satisfied": h_hetero,
        "discriminating_scenario": discriminator,
        "discriminating_verdict": verdicts.get(discriminator),
        "outcome": outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    config = load_json(arguments.config)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    summary = load_json(arguments.summary)

    rule = config["decision_rule"]
    results = {
        scenario["name"]: evaluate_scenario(
            scenario["name"],
            summary[scenario["name"]],
            str(rule["candidate_method"]),
            str(rule["grid_prefix"]),
            int(rule["max_dropped_seeds"]),
        )
        for scenario in config["scenarios"]
    }

    decision = decide(config, results)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)

    print(f"outcome: {decision['outcome']}")
    for name, verdict in decision["verdicts"].items():
        entry = results[name]["primary"]["delta_frontier"]
        if entry is None:
            print(f"  {name:<46}{verdict}")
        else:
            print(
                f"  {name:<46}{verdict:<13}"
                f"{entry['mean']:+.5f} [{entry['ci_lower']:+.5f}, "
                f"{entry['ci_upper']:+.5f}]  dropped="
                f"{results[name]['primary']['dropped_seeds']}"
            )


if __name__ == "__main__":
    main()
