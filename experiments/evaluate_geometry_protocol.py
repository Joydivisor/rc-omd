"""Apply the frozen geometry-v1 selection and Go/No-Go rules.

Two modes, matching the protocol's two evaluation phases:

* ``--mode dev`` applies the selection rule to a development sweep and emits
  the winning ``(lambda, mu)``. It never emits a Go/No-Go decision.
* ``--mode test`` applies the pre-declared decision rule to the diagnostic and
  held-out scenarios, using the already-frozen ``(lambda, mu)``.

Statistics follow `pareto-v1-2026-08-14`: paired per-seed differences with
two-sided 95% t-confidence bounds, and a log-scale bound on the distractor-KL
ratio. The power clause differs from `pareto-v1` deliberately, per
`docs/GEOMETRY_V1_PROTOCOL.md`: both conditions here are conservative under
widening, so a wide interval can never manufacture a pass, and the clause
therefore applies only to failures, to stop an imprecise failure being read as
evidence of absence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "geometry-v1-2026-08-15"

# scipy.stats.t.ppf(0.975, df). The protocol fixes 20 seeds per cell and allows
# at most 2 degenerate seeds to be dropped from the KL test, so df is in
# {17, 18, 19}.
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
            "geometry-v1 is frozen at 20 seeds/cell with at most 2 KL seeds "
            "droppable, so df must be in {17, 18, 19}"
        )
    mean = float(np.mean(values))
    half_width = _T_CRITICAL_95[df] * float(np.std(values, ddof=1)) / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "half_width": half_width,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def _paired(baseline: dict[str, Any], candidate: dict[str, Any], key: str,
            scenario: str) -> tuple[np.ndarray, np.ndarray]:
    if baseline["seeds"] != candidate["seeds"]:
        raise ValueError(
            f"seed mismatch in scenario {scenario}; pairing assumption violated"
        )
    return (
        np.asarray(baseline[key], dtype=float),
        np.asarray(candidate[key], dtype=float),
    )


def evaluate_scenario(
    scenario: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Return the pass / fail / inconclusive verdict for one scenario."""

    margin = float(rule["auc_non_inferiority_margin"])
    bound = float(rule["distractor_kl_ratio_bound"])
    degenerate = float(rule["degenerate_kl_threshold"])
    max_dropped = int(rule["max_dropped_kl_seeds"])
    auc_floor = float(rule["auc_precision_floor"])
    log_kl_floor = float(rule["log_kl_precision_floor"])

    base_auc, cand_auc = _paired(baseline, candidate, "success_auc_per_seed", scenario)
    auc_ci = t_ci_95(cand_auc - base_auc)
    auc_pass = auc_ci["ci_lower"] > -margin

    base_kl, cand_kl = _paired(
        baseline, candidate, "cumulative_distractor_kl_per_seed", scenario
    )
    valid = base_kl >= degenerate
    dropped = int(np.sum(~valid))
    kl_report: dict[str, Any] = {"dropped_seeds": dropped}
    if dropped > max_dropped:
        # Too much of the sample discarded to adjudicate the bound at all.
        kl_pass = False
        kl_degenerate = True
        kl_report["degenerate"] = True
    else:
        kl_degenerate = False
        log_ratio_ci = t_ci_95(np.log(cand_kl[valid] / base_kl[valid]))
        upper_ratio = math.exp(log_ratio_ci["ci_upper"])
        kl_pass = upper_ratio <= bound
        kl_report.update(
            {"degenerate": False, "log_ratio": log_ratio_ci, "upper_ratio": upper_ratio}
        )

    if auc_pass and kl_pass:
        status = "pass"
    else:
        # Power clause: only failures can be downgraded to inconclusive, and
        # only when the measurement was too imprecise to have detected the
        # effect under test.
        underpowered = False
        if not auc_pass and auc_ci["half_width"] > auc_floor:
            underpowered = True
        if not kl_pass:
            if kl_degenerate:
                underpowered = True
            elif kl_report["log_ratio"]["half_width"] > log_kl_floor:
                underpowered = True
        status = "inconclusive" if underpowered else "fail"

    return {
        "scenario": scenario,
        "auc": {"d_auc": auc_ci, "auc_pass": auc_pass},
        "kl": kl_report | {"kl_pass": kl_pass},
        "status": status,
    }


def check_invariant(
    scenario: str, methods: dict[str, Any], names: list[str], tolerance: float
) -> dict[str, Any]:
    """Complete aliasing forces equal critical and distractor realized KL.

    This is a theorem about the parameterization, so a violation is an
    implementation defect and the protocol requires halting rather than
    reporting a result.
    """

    per_method = {}
    holds = True
    for name in names:
        entry = methods[name]
        critical = np.asarray(entry["cumulative_critical_kl_per_seed"], dtype=float)
        distractor = np.asarray(
            entry["cumulative_distractor_kl_per_seed"], dtype=float
        )
        worst = float(np.abs(critical - distractor).max())
        ok = worst <= tolerance
        holds = holds and ok
        per_method[name] = {"max_abs_difference": worst, "holds": ok}
    return {"scenario": scenario, "methods": per_method, "invariant_holds": holds}


def run_selection(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    rule = config["selection"]
    baseline_name = str(rule["baseline_method"])
    regression = str(rule["regression_scenario"])
    selection_scenarios = list(rule["selection_scenarios"])

    invariants = [
        check_invariant(
            scenario,
            summary[scenario],
            [point["name"] for point in rule["grid"]],
            float(rule["invariant_tolerance"]),
        )
        for scenario in rule["invariant_scenarios"]
    ]
    invariant_holds = all(item["invariant_holds"] for item in invariants)

    candidates = []
    for point in list(rule["grid"]) + [
        {"name": str(rule["negative_check_method"]), "lambda": None, "mu": None}
    ]:
        name = point["name"]
        scenarios = {
            scenario: evaluate_scenario(
                scenario, summary[scenario][baseline_name], summary[scenario][name], rule
            )
            for scenario in selection_scenarios
        }
        passes = sum(1 for r in scenarios.values() if r["status"] == "pass")
        upper_bounds = [
            r["kl"]["upper_ratio"]
            for r in scenarios.values()
            if not r["kl"].get("degenerate", False)
        ]
        candidates.append(
            {
                "name": name,
                "lambda": point["lambda"],
                "mu": point["mu"],
                "is_negative_check": point["lambda"] is None,
                "scenarios": scenarios,
                "pass_count": passes,
                "mean_kl_upper_bound": (
                    float(np.mean(upper_bounds)) if upper_bounds else float("inf")
                ),
                "passes_regression_gate": scenarios[regression]["status"] == "pass",
            }
        )

    grid_only = [c for c in candidates if not c["is_negative_check"]]
    eligible = [c for c in grid_only if c["passes_regression_gate"]]

    selected = None
    if eligible:
        # Deterministic: pass count desc, then mean KL upper bound asc, then
        # lambda asc, then mu asc. Grid order is the final tie-break, and the
        # grid is emitted lambda-major so list position encodes it.
        order = {c["name"]: i for i, c in enumerate(grid_only)}
        selected = sorted(
            eligible,
            key=lambda c: (
                -c["pass_count"],
                c["mean_kl_upper_bound"],
                c["lambda"],
                c["mu"],
                order[c["name"]],
            ),
        )[0]

    negative = next(c for c in candidates if c["is_negative_check"])
    best_pass = max((c["pass_count"] for c in eligible), default=-1)
    negative_check_warning = negative["pass_count"] > best_pass

    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "development",
        "invariants": invariants,
        "invariant_holds": invariant_holds,
        "candidates": candidates,
        "eligible_count": len(eligible),
        "selected": (
            None
            if selected is None
            else {
                "method": selected["name"],
                "lambda": selected["lambda"],
                "mu": selected["mu"],
                "pass_count": selected["pass_count"],
                "mean_kl_upper_bound": selected["mean_kl_upper_bound"],
            }
        ),
        "negative_check": {
            "method": negative["name"],
            "pass_count": negative["pass_count"],
            "outperformed_grid": negative_check_warning,
        },
        "outcome": (
            "HALT_INVARIANT_VIOLATED"
            if not invariant_holds
            else "NO-GO_NO_ELIGIBLE_CANDIDATE"
            if selected is None
            else "SELECTED"
        ),
    }


def run_decision(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    rule = config["decision_rule"]
    baseline_name = str(rule["baseline_method"])
    candidate_name = str(rule["candidate_method"])
    required = list(rule["required_scenarios"])

    invariants = [
        check_invariant(
            scenario,
            summary[scenario],
            [candidate_name],
            float(rule["invariant_tolerance"]),
        )
        for scenario in rule["invariant_scenarios"]
    ]
    invariant_holds = all(item["invariant_holds"] for item in invariants)

    scenarios = {
        scenario: evaluate_scenario(
            scenario,
            summary[scenario][baseline_name],
            summary[scenario][candidate_name],
            rule,
        )
        for scenario in required
    }
    statuses = [r["status"] for r in scenarios.values()]
    any_inconclusive = "inconclusive" in statuses
    all_pass = all(s == "pass" for s in statuses)

    if not invariant_holds:
        decision = "HALT_INVARIANT_VIOLATED"
    elif any_inconclusive:
        decision = "INCONCLUSIVE"
    elif all_pass:
        decision = "GO"
    else:
        decision = "NO-GO"

    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "test",
        "frozen_thresholds": rule,
        "scenario_results": scenarios,
        "invariants": invariants,
        "invariant_holds": invariant_holds,
        "required_scenarios": required,
        "pass_count": sum(1 for s in statuses if s == "pass"),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "test"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    config = load_json(arguments.config)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    summary = load_json(arguments.summary)

    if arguments.mode == "dev":
        result = run_selection(config, summary)
    else:
        result = run_decision(config, summary)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
