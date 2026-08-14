"""Apply the frozen pareto-v1 Go/No-Go rule to a generated summary.

Implements the statistical rules in `docs/PARETO_V1_PROTOCOL.md`: paired
per-seed differences, a deterministic AUC-first matching rule between the
Uniform step grid and the Online step grid, a log-scale confidence bound on
the distractor-KL ratio at the matched step, and a coverage-count decision
rule with an anti-shrinkage and a power-failure clause.

Two implementation details are left underspecified by the protocol text and
are pinned down here, stated explicitly so the choice is auditable rather than
silently baked in:

1. Power-failure check ("the 95% CI half-width on d_AUC" at each tested
   Uniform grid point). Read as: the CI from the comparison that decided
   coverage for that point -- the matched Online step if the point is
   covered or matched-but-KL-failed, otherwise the grid maximum (the last
   step attempted before the point was declared uncovered).
2. "Median runtime ratio at the matched step pairs." Read as: pool the
   per-seed runtime ratios (online / uniform, paired by seed) across every
   Uniform point that found a matched Online step in the scenario, then take
   one median over the pooled values.

The two-sided 95% t-critical values below are hardcoded rather than pulled
from scipy.stats, because this protocol is frozen at 20 seeds/cell (df=19)
for the AUC test, and drops at most 2 degenerate seeds for the KL test
(df in {17, 18, 19}); a new project dependency is not needed for three
constants, matching this repo's numpy-only footprint
(`scipy.stats.t.ppf(0.975, df=df)` for df in {17, 18, 19}). Any change to the
seed count requires a new protocol ID under the protocol's own execution
discipline, at which point new constants would be required anyway.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "pareto-v1-2026-08-14"

# scipy.stats.t.ppf(0.975, df=df), tabulated for the only df values this
# frozen protocol can produce.
_T_CRITICAL_95 = {
    17: 2.1098155778333156,
    18: 2.1009220402410382,
    19: 2.0930240544083087,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def method_name(prefix: str, step: float) -> str:
    return f"{prefix}{round(step * 100):03d}"


def t_ci_95(values: np.ndarray) -> dict[str, float]:
    """Two-sided 95% t-confidence interval on the mean of `values`."""

    n = int(values.shape[0])
    df = n - 1
    if df not in _T_CRITICAL_95:
        raise ValueError(
            f"no tabulated two-sided 95% t-critical value for df={df}; "
            "pareto-v1 is frozen at 20 seeds/cell with at most 2 KL seeds "
            "droppable, so df must be in {17, 18, 19}"
        )
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = _T_CRITICAL_95[df] * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "half_width": half_width,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def auc_comparison(
    uniform_auc: np.ndarray, online_auc: np.ndarray, online_step: float, margin: float
) -> dict[str, Any]:
    diffs = online_auc - uniform_auc
    ci = t_ci_95(diffs)
    return {
        "online_step": online_step,
        "d_auc": ci,
        "auc_pass": ci["ci_lower"] > -margin,
    }


def kl_comparison(
    uniform_kl: np.ndarray,
    online_kl: np.ndarray,
    bound: float,
    degenerate_threshold: float,
    max_dropped: int,
) -> dict[str, Any]:
    valid_mask = uniform_kl >= degenerate_threshold
    dropped = int(np.sum(~valid_mask))
    if dropped > max_dropped:
        return {
            "dropped_seeds": dropped,
            "inconclusive": True,
            "kl_pass": False,
        }
    log_ratio = np.log(online_kl[valid_mask] / uniform_kl[valid_mask])
    ci = t_ci_95(log_ratio)
    upper_ratio = math.exp(ci["ci_upper"])
    return {
        "dropped_seeds": dropped,
        "inconclusive": False,
        "log_ratio": ci,
        "upper_ratio": upper_ratio,
        "kl_pass": upper_ratio <= bound,
    }


def evaluate_scenario(
    scenario_name: str,
    methods_summary: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    step_grid = [float(step) for step in rule["step_grid"]]
    uniform_prefix = str(rule["uniform_method_prefix"])
    online_prefix = str(rule["online_method_prefix"])
    margin = float(rule["auc_non_inferiority_margin"])
    bound = float(rule["distractor_kl_ratio_bound"])
    degenerate_threshold = float(rule["degenerate_kl_threshold"])
    max_dropped = int(rule["max_dropped_kl_seeds"])
    power_fraction = float(rule["power_failure_ci_half_width_fraction"])

    points: list[dict[str, Any]] = []
    underpowered_count = 0
    pooled_runtime_ratios: list[float] = []

    for uniform_step in step_grid:
        uniform_name = method_name(uniform_prefix, uniform_step)
        uniform_data = methods_summary[uniform_name]
        uniform_seeds = uniform_data["seeds"]
        uniform_auc = np.asarray(uniform_data["success_auc_per_seed"], dtype=float)
        uniform_kl = np.asarray(
            uniform_data["cumulative_distractor_kl_per_seed"], dtype=float
        )
        uniform_runtime = np.asarray(
            uniform_data["runtime_seconds_per_seed"], dtype=float
        )

        matched_comparison: dict[str, Any] | None = None
        last_tested_comparison: dict[str, Any] | None = None
        for online_step in step_grid:
            online_name = method_name(online_prefix, online_step)
            online_data = methods_summary[online_name]
            if online_data["seeds"] != uniform_seeds:
                raise ValueError(
                    f"seed mismatch between {uniform_name} and {online_name} "
                    f"in scenario {scenario_name}; pairing assumption violated"
                )
            online_auc = np.asarray(online_data["success_auc_per_seed"], dtype=float)
            comparison = auc_comparison(uniform_auc, online_auc, online_step, margin)
            last_tested_comparison = comparison
            if comparison["auc_pass"]:
                matched_comparison = comparison
                break

        if matched_comparison is None:
            decision_comparison = last_tested_comparison
            matched_step = None
            kl_result = None
            covered = False
        else:
            decision_comparison = matched_comparison
            matched_step = matched_comparison["online_step"]
            online_name = method_name(online_prefix, matched_step)
            online_data = methods_summary[online_name]
            online_kl = np.asarray(
                online_data["cumulative_distractor_kl_per_seed"], dtype=float
            )
            kl_result = kl_comparison(
                uniform_kl, online_kl, bound, degenerate_threshold, max_dropped
            )
            covered = bool(kl_result["kl_pass"])

            online_runtime = np.asarray(
                online_data["runtime_seconds_per_seed"], dtype=float
            )
            pooled_runtime_ratios.extend(
                (online_runtime / uniform_runtime).tolist()
            )

        assert decision_comparison is not None
        if decision_comparison["d_auc"]["half_width"] > margin:
            underpowered_count += 1

        points.append(
            {
                "uniform_step": uniform_step,
                "matched_online_step": matched_step,
                "auc": decision_comparison,
                "kl": kl_result,
                "covered": covered,
                "frontier_truncated": matched_step is not None
                and matched_step == step_grid[-1],
            }
        )

    total_points = len(step_grid)
    covered_count = sum(1 for point in points if point["covered"])
    underpowered = underpowered_count > power_fraction * total_points
    if underpowered:
        status = "inconclusive"
    elif covered_count >= int(rule["required_covered_points_per_scenario"]):
        status = "pass"
    else:
        status = "fail"

    median_runtime_ratio = (
        float(np.median(pooled_runtime_ratios)) if pooled_runtime_ratios else None
    )
    runtime_pass = (
        median_runtime_ratio is not None
        and median_runtime_ratio <= float(rule["max_runtime_ratio"])
    )

    return {
        "points": points,
        "covered_count": covered_count,
        "total_points": total_points,
        "underpowered_point_count": underpowered_count,
        "status": status,
        "median_runtime_ratio": median_runtime_ratio,
        "runtime_pass": runtime_pass,
    }


def evaluate(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    expected_scenarios = {scenario["name"] for scenario in config["scenarios"]}
    if set(summary) != expected_scenarios:
        raise ValueError("summary scenarios do not match the frozen protocol")

    rule = config["decision_rule"]
    scenario_reports = {
        scenario: evaluate_scenario(scenario, summary[scenario], rule)
        for scenario in sorted(expected_scenarios)
    }

    scenario_pass_count = sum(
        1 for report in scenario_reports.values() if report["status"] == "pass"
    )
    any_inconclusive = any(
        report["status"] == "inconclusive" for report in scenario_reports.values()
    )
    decision = (
        "GO"
        if scenario_pass_count >= int(rule["required_scenario_passes"])
        and not any_inconclusive
        else "NO-GO"
    )
    systems_feasibility_decision = (
        "PASS"
        if all(report["runtime_pass"] for report in scenario_reports.values())
        else "FAIL"
    )

    return {
        "protocol_id": PROTOCOL_ID,
        "frozen_thresholds": rule,
        "scenario_reports": scenario_reports,
        "scenario_pass_count": scenario_pass_count,
        "scenario_inconclusive": any_inconclusive,
        "decision": decision,
        "systems_feasibility_decision": systems_feasibility_decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate(load_json(arguments.config), load_json(arguments.summary))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
