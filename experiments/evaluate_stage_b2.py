"""Apply the frozen `stage-b2-estimator-2026-08-19` gates and decision rules.

Three modes:

* ``--mode calibrate`` runs the blocking, tie-aware calibration gate on every
  menu entry against the linear head.
* ``--mode select`` computes each admissible variant's correlation on the Stage
  B **discovery** instances and emits the selected estimator.
* ``--mode confirm`` evaluates the frozen estimator once on the fresh
  confirmation family.

The calibration gate is pairwise rather than Spearman-based: the discrete index
is heavily tied, and a rank correlation penalises the arbitrary ordering a
continuous estimator imposes inside a tie group where the ground truth
expresses no order. See the amendment in
`docs/STAGE_B2_ESTIMATOR_PROTOCOL.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.evaluate_geometry_v3 import (
    bootstrap_rho_interval,
    scenario_metric,
    spearman,
)
from experiments.run_reliability_diagnostics import build_algorithm
from experiments.stage_b2_estimators import (
    ESTIMATORS,
    PROTOCOL_ID,
    REFUTED_ESTIMATOR,
    alpha_from_jacobians,
    measure_all_over_seeds,
)
from experiments.stage_b_alpha import position_jacobians


def _signs(scenario: dict[str, Any]) -> np.ndarray:
    critical = {int(k) for k in scenario["critical_positions"]}
    return np.array(
        [1.0 if k in critical else -1.0 for k in range(int(scenario["horizon"]))],
        dtype=np.float64,
    )


def discordant_pairs(measured: np.ndarray, truth: np.ndarray) -> int:
    """Non-tied pairs the estimator orders against the ground truth."""

    bad = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if truth[i] == truth[j]:
                continue
            if (truth[i] < truth[j]) != (measured[i] < measured[j]):
                bad += 1
    return bad


def run_calibration(stage_b_config: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(stage_b_config["methods"]["uniform_eta075000"])
    baseline["algorithm"] = "projected_group_omd"
    baseline.pop("mlp_hidden", None)

    instances = [
        s for s in stage_b_config["scenarios"]
        if float(s.get("feature_noise", 0.0)) == 0.0
    ]
    truth = np.array(
        [stage_b_config["discrete_alpha"][s["name"]] for s in instances]
    )
    measured: dict[str, list[float]] = {name: [] for name in ESTIMATORS}
    for scenario in instances:
        algorithm = build_algorithm("calib", baseline, scenario, 0)
        raw = position_jacobians(algorithm)
        context = {
            "head": getattr(algorithm, "head", None),
            "n_actions": int(algorithm.n_actions),
            "policy": algorithm.policy,
        }
        signs = _signs(scenario)
        for name, transform in ESTIMATORS.items():
            measured[name].append(alpha_from_jacobians(transform(raw, context), signs))

    table: dict[str, Any] = {}
    for name in ESTIMATORS:
        values = np.asarray(measured[name])
        finite = bool(np.all(np.isfinite(values)))
        bad = discordant_pairs(values, truth) if finite else -1
        table[name] = {
            "finite": finite,
            "discordant_non_tied_pairs": bad,
            "exact_reproduction": bool(
                finite and np.max(np.abs(values - truth)) < 1e-9
            ),
            "spearman_for_reference": spearman(values, truth) if finite else None,
            "admissible": bool(finite and bad == 0),
        }
    n_pairs = sum(
        1 for i in range(len(truth)) for j in range(i + 1, len(truth))
        if truth[i] != truth[j]
    )
    return {
        "protocol_id": PROTOCOL_ID, "phase": "calibration",
        "n_instances": len(instances), "n_non_tied_pairs": n_pairs,
        "distinct_discrete_values": int(len(set(np.round(truth, 9).tolist()))),
        "per_estimator": table,
        "admissible": [n for n, e in table.items() if e["admissible"]],
        "outcome": (
            "PASS" if any(e["admissible"] for e in table.values())
            else "HALT_NO_ADMISSIBLE_ESTIMATOR"
        ),
    }


def _alphas(config, names, by_name, seeds_key="seeds"):
    baseline = config["methods"]["uniform_eta075000"]
    interval = int(config["training"]["evaluation_interval"])
    return {
        name: measure_all_over_seeds(
            by_name[name], baseline,
            [int(s) for s in by_name[name][seeds_key]], interval,
        )
        for name in names
    }


def run_selection(stage_b_config, stage_b_summary, calibration) -> dict[str, Any]:
    rule = {"candidate_method": "rwp_eta125_lam3", "grid_prefix": "uniform_eta",
            "max_dropped_seeds": 2}
    names = list(stage_b_config["split"]["discovery"])
    by_name = {s["name"]: s for s in stage_b_config["scenarios"]}

    metrics = {
        n: scenario_metric(stage_b_summary[n], rule["candidate_method"],
                           rule["grid_prefix"], rule["max_dropped_seeds"])
        for n in names
    }
    alphas = _alphas(stage_b_config, names, by_name)
    usable = [n for n in names if metrics[n]["usable"]]
    response = np.array([metrics[n]["delta_corrected"] for n in usable])

    admissible = calibration["admissible"]
    ranked = []
    for position, name in enumerate(ESTIMATORS):
        if name not in admissible:
            continue
        values = np.array([alphas[n][name] for n in usable])
        rho = spearman(values, response) if np.all(np.isfinite(values)) else 0.0
        ranked.append({"estimator": name, "rho": rho, "abs_rho": abs(rho),
                       "menu_position": position})
    ranked.sort(key=lambda e: (-e["abs_rho"], e["menu_position"]))

    threshold = 0.50
    best = ranked[0] if ranked else None
    selected = best if best and best["abs_rho"] >= threshold else None
    return {
        "protocol_id": PROTOCOL_ID, "phase": "selection",
        "instances": names, "usable_instances": usable,
        "alpha": alphas, "metrics": metrics,
        "correlations": ranked, "selection_threshold_abs_rho": threshold,
        "selected": selected,
        "refuted_estimator_selected": bool(
            selected and selected["estimator"] == REFUTED_ESTIMATOR
        ),
        "outcome": "SELECTED" if selected else "NO_ESTIMATOR_FOUND",
    }


def run_confirmation(config, summary, estimator: str) -> dict[str, Any]:
    rule = config["decision_rule"]
    names = [s["name"] for s in config["scenarios"]]
    by_name = {s["name"]: s for s in config["scenarios"]}
    metrics = {
        n: scenario_metric(summary[n], str(rule["candidate_method"]),
                           str(rule["grid_prefix"]), int(rule["max_dropped_seeds"]))
        for n in names
    }
    alphas = _alphas(config, names, by_name)
    usable = [
        n for n in names
        if metrics[n]["usable"] and np.isfinite(alphas[n][estimator])
    ]
    x = np.array([alphas[n][estimator] for n in usable])
    y = np.array([metrics[n]["delta_corrected"] for n in usable])

    interval = bootstrap_rho_interval(
        x, y, int(rule["bootstrap_resamples"]), int(rule["bootstrap_seed"])
    )
    bound = float(rule["confirmation_magnitude_bound"])
    sign_ok = interval["rho"] < 0
    clears = interval["ci_upper"] < -bound
    confirmed = sign_ok and clears
    return {
        "protocol_id": PROTOCOL_ID, "phase": "confirmation",
        "estimator": estimator, "instances": names, "usable_instances": usable,
        "alpha": alphas, "metrics": metrics, "spearman": interval,
        "required_magnitude_bound": bound,
        "sign_matches_expected": sign_ok, "magnitude_clears_bound": clears,
        "outcome": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
        "consequence": (
            "Aliasing geometry is measurable under non-linearity; the Qwen "
            "pre-flight gate is reinstated as an evidence-backed candidate gate"
            if confirmed else
            "Direct progression to Qwen remains PROHIBITED; the E4-style account "
            "of the Stage B failure is wrong or incomplete"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "select", "confirm"],
                        required=True)
    parser.add_argument("--stage-b-config", type=Path,
                        default=Path("configs/stage_b_mlp.json"))
    parser.add_argument("--stage-b-summary", type=Path,
                        default=Path("results/stage_b_mlp/summary.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage_b2.json"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--estimator")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    stage_b = json.loads(arguments.stage_b_config.read_text(encoding="utf-8"))

    if arguments.mode == "calibrate":
        result = run_calibration(stage_b)
        print(f"outcome: {result['outcome']}  "
              f"({result['n_non_tied_pairs']} non-tied pairs, "
              f"{result['distinct_discrete_values']} distinct values)")
        for name, entry in result["per_estimator"].items():
            print(f"  {name:<30}discordant {entry['discordant_non_tied_pairs']:>3}  "
                  f"exact={str(entry['exact_reproduction']):<5} "
                  f"{'ADMISSIBLE' if entry['admissible'] else 'DISQUALIFIED'}")
    elif arguments.mode == "select":
        calibration = json.loads(arguments.calibration.read_text(encoding="utf-8"))
        summary = json.loads(arguments.stage_b_summary.read_text(encoding="utf-8"))
        result = run_selection(stage_b, summary, calibration)
        print(f"outcome: {result['outcome']}")
        for entry in result["correlations"]:
            print(f"  {entry['estimator']:<30}rho {entry['rho']:+.4f}")
        if result["selected"]:
            print(f"  selected: {result['selected']['estimator']}")
    else:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
        result = run_confirmation(config, summary, arguments.estimator)
        interval = result["spearman"]
        print(f"outcome: {result['outcome']}")
        print(f"  {result['estimator']}: rho {interval['rho']:+.4f} "
              f"[{interval['ci_lower']:+.4f}, {interval['ci_upper']:+.4f}]")
        print(f"  {result['consequence']}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
