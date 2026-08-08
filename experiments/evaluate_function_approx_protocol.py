"""Apply the frozen function-approx-v1 Go/No-Go rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "function-approx-v1-2026-08-09"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    expected_scenarios = {scenario["name"] for scenario in config["scenarios"]}
    if set(summary) != expected_scenarios:
        raise ValueError("summary scenarios do not match the frozen protocol")

    rule = config["decision_rule"]
    uniform_name = str(rule["primary_uniform"])
    online_name = str(rule["primary_online"])
    scenario_results: dict[str, Any] = {}
    scenario_passes = 0
    runtime_passes = 0
    for scenario in sorted(expected_scenarios):
        uniform = summary[scenario][uniform_name]
        online = summary[scenario][online_name]
        auc_difference = float(online["success_auc_mean"]) - float(
            uniform["success_auc_mean"]
        )
        uniform_kl = float(uniform["cumulative_distractor_kl_mean"])
        online_kl = float(online["cumulative_distractor_kl_mean"])
        kl_ratio = online_kl / uniform_kl
        runtime_ratio = float(online["runtime_seconds_mean"]) / float(
            uniform["runtime_seconds_mean"]
        )
        numerical_tolerance = 1e-12
        auc_pass = (
            auc_difference
            >= -float(rule["max_auc_deficit"]) - numerical_tolerance
        )
        kl_pass = (
            kl_ratio
            <= float(rule["max_distractor_kl_ratio"]) + numerical_tolerance
        )
        runtime_pass = (
            runtime_ratio <= float(rule["max_runtime_ratio"]) + numerical_tolerance
        )
        scenario_pass = auc_pass and kl_pass
        scenario_passes += int(scenario_pass)
        runtime_passes += int(runtime_pass)
        scenario_results[scenario] = {
            "uniform_auc": float(uniform["success_auc_mean"]),
            "online_auc": float(online["success_auc_mean"]),
            "auc_difference": auc_difference,
            "uniform_distractor_kl": uniform_kl,
            "online_distractor_kl": online_kl,
            "distractor_kl_ratio": kl_ratio,
            "runtime_ratio": runtime_ratio,
            "auc_pass": auc_pass,
            "kl_pass": kl_pass,
            "runtime_pass": runtime_pass,
            "scenario_pass": scenario_pass,
        }

    return {
        "protocol_id": PROTOCOL_ID,
        "primary_pair": {"uniform": uniform_name, "online": online_name},
        "frozen_thresholds": rule,
        "scenario_results": scenario_results,
        "scenario_pass_count": scenario_passes,
        "runtime_pass_count": runtime_passes,
        "transfer_decision": (
            "GO"
            if scenario_passes >= int(rule["required_scenario_passes"])
            else "NO-GO"
        ),
        "systems_feasibility_decision": (
            "PASS"
            if runtime_passes >= int(rule["required_runtime_passes"])
            else "FAIL"
        ),
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
