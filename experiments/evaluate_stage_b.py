"""Apply the frozen `stage-b-mlp-2026-08-19` gates and decision rules.

Three modes:

* ``--mode calibrate`` runs the blocking calibration gate: the continuous
  ``alpha`` estimator, applied to the **linear** head, must reproduce the
  discrete aliasing index. It gates the instrument, not the hypothesis.
* ``--mode discovery`` reports the correlation on the discovery instances.
  Reported, never decisive.
* ``--mode confirmation`` executes the held-out test once.

Metric, matching, and bias correction are inherited from geometry-v2/v3
unchanged. The confirmation criterion is direction-agnostic per E11, and both
its threshold and its expected sign are inherited from geometry-v3 rather than
chosen here.
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
from experiments.stage_b_alpha import (
    alpha_from_jacobians,
    measure_alpha_over_seeds,
    position_jacobians,
)

PROTOCOL_ID = "stage-b-mlp-2026-08-19"


def _signs(scenario: dict[str, Any]) -> np.ndarray:
    critical = {int(k) for k in scenario["critical_positions"]}
    return np.array(
        [1.0 if k in critical else -1.0 for k in range(int(scenario["horizon"]))],
        dtype=np.float64,
    )


def run_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Estimator must reproduce discrete alpha under the LINEAR head."""

    tolerance = float(config["decision_rule"]["calibration_tolerance"])
    baseline = dict(config["methods"][config["decision_rule"]["baseline_alpha_method"]])
    baseline["algorithm"] = "projected_group_omd"  # linear head
    baseline.pop("mlp_hidden", None)

    per_scenario: dict[str, dict[str, float]] = {}
    worst = 0.0
    for scenario in config["scenarios"]:
        if float(scenario.get("feature_noise", 0.0)) != 0.0:
            continue  # the anchor is defined on unperturbed features only
        algorithm = build_algorithm("calib", baseline, scenario, 0)
        measured = alpha_from_jacobians(
            position_jacobians(algorithm), _signs(scenario)
        )
        expected = float(config["discrete_alpha"][scenario["name"]])
        error = abs(measured - expected)
        worst = max(worst, error)
        per_scenario[scenario["name"]] = {
            "measured": measured, "discrete": expected, "abs_error": error
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "phase": "calibration",
        "per_scenario": per_scenario,
        "max_abs_error": worst,
        "tolerance": tolerance,
        "outcome": "PASS" if worst <= tolerance else "HALT_ESTIMATOR_MISCALIBRATED",
    }


def measure(config: dict[str, Any], summary: dict[str, Any], names: list[str]):
    """Frontier metric and measured alpha for each named instance."""

    rule = config["decision_rule"]
    baseline = config["methods"][rule["baseline_alpha_method"]]
    by_name = {s["name"]: s for s in config["scenarios"]}
    interval = int(config["training"]["evaluation_interval"])

    metrics: dict[str, Any] = {}
    alphas: dict[str, Any] = {}
    for name in names:
        metrics[name] = scenario_metric(
            summary[name], str(rule["candidate_method"]),
            str(rule["grid_prefix"]), int(rule["max_dropped_seeds"]),
        )
        scenario = by_name[name]
        alphas[name] = measure_alpha_over_seeds(
            scenario, baseline, [int(s) for s in scenario["seeds"]], interval
        )
    return metrics, alphas


def _paired(metrics, alphas, names):
    usable = [n for n in names if metrics[n]["usable"]
              and np.isfinite(alphas[n]["alpha"])]
    x = np.array([alphas[n]["alpha"] for n in usable])
    y = np.array([metrics[n]["delta_corrected"] for n in usable])
    return usable, x, y


def run_discovery(config, summary) -> dict[str, Any]:
    names = list(config["split"]["discovery"])
    metrics, alphas = measure(config, summary, names)
    usable, x, y = _paired(metrics, alphas, names)
    by_name = {s["name"]: s for s in config["scenarios"]}
    per_eps = {}
    for eps in config["eps_levels"]:
        subset = [n for n in usable if by_name[n]["feature_noise"] == eps]
        if len(subset) >= 4:
            per_eps[str(eps)] = {
                "n": len(subset),
                "rho": spearman(
                    np.array([alphas[n]["alpha"] for n in subset]),
                    np.array([metrics[n]["delta_corrected"] for n in subset]),
                ),
            }
    return {
        "protocol_id": PROTOCOL_ID, "phase": "discovery",
        "instances": names, "usable_instances": usable,
        "metrics": metrics, "alpha": alphas,
        "rho": spearman(x, y) if len(usable) > 2 else float("nan"),
        "rho_within_eps": per_eps,
    }


def run_confirmation(config, summary) -> dict[str, Any]:
    rule = config["decision_rule"]
    names = list(config["split"]["confirmation"])
    metrics, alphas = measure(config, summary, names)
    usable, x, y = _paired(metrics, alphas, names)

    interval = bootstrap_rho_interval(
        x, y, int(rule["bootstrap_resamples"]), int(rule["bootstrap_seed"])
    )
    sign = int(rule["expected_sign"])
    bound = float(rule["confirmation_magnitude_bound"])
    sign_matches = interval["rho"] < 0 if sign < 0 else interval["rho"] > 0
    clears = (
        interval["ci_upper"] < -bound if sign < 0 else interval["ci_lower"] > bound
    )
    confirmed = sign_matches and clears

    by_name = {s["name"]: s for s in config["scenarios"]}
    per_eps = {}
    for eps in config["eps_levels"]:
        subset = [n for n in usable if by_name[n]["feature_noise"] == eps]
        if len(subset) >= 4:
            per_eps[str(eps)] = {
                "n": len(subset),
                "rho": spearman(
                    np.array([alphas[n]["alpha"] for n in subset]),
                    np.array([metrics[n]["delta_corrected"] for n in subset]),
                ),
            }

    return {
        "protocol_id": PROTOCOL_ID, "phase": "confirmation",
        "instances": names, "usable_instances": usable,
        "metrics": metrics, "alpha": alphas,
        "spearman": interval,
        "expected_sign": sign, "required_magnitude_bound": bound,
        "sign_matches_expected": sign_matches, "magnitude_clears_bound": clears,
        "rho_within_eps": per_eps,
        "outcome": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
        "consequence": (
            "Qwen pre-flight gate reinstated as an evidence-backed candidate gate"
            if confirmed else
            "Direct progression to Qwen is PROHIBITED; mechanistic investigation "
            "resumes on synthetic scenarios"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["calibrate", "discovery", "confirmation"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")

    if arguments.mode == "calibrate":
        result = run_calibration(config)
        print(f"outcome: {result['outcome']}  "
              f"max abs error {result['max_abs_error']:.3e} "
              f"vs tolerance {result['tolerance']:.0e}")
    else:
        if not arguments.summary:
            raise SystemExit("--summary is required in this mode")
        summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
        if arguments.mode == "discovery":
            result = run_discovery(config, summary)
            print(f"discovery rho {result['rho']:+.4f} on "
                  f"{len(result['usable_instances'])} instances")
            for eps, entry in result["rho_within_eps"].items():
                print(f"  eps={eps}: rho {entry['rho']:+.4f} (n={entry['n']})")
        else:
            result = run_confirmation(config, summary)
            spear = result["spearman"]
            print(f"outcome: {result['outcome']}")
            print(f"  rho {spear['rho']:+.4f} "
                  f"[{spear['ci_lower']:+.4f}, {spear['ci_upper']:+.4f}] "
                  f"vs magnitude bound {result['required_magnitude_bound']}")
            for eps, entry in result["rho_within_eps"].items():
                print(f"  eps={eps}: rho {entry['rho']:+.4f} (n={entry['n']})")
            print(f"  {result['consequence']}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
