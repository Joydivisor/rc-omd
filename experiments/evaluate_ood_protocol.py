"""Apply the frozen ood-v1 Go/No-Go rule to a generated summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "ood-v1-2026-08-08"
PRIMARY_UNIFORM = "uniform_eta075"
PRIMARY_ONLINE = "online_eta125"
MAX_AUC_DEFICIT = 0.01
MAX_DISTRACTOR_KL_RATIO = 0.5
MAX_RUNTIME_RATIO = 1.5
REQUIRED_SCENARIO_PASSES = 3


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"expected protocol_id {PROTOCOL_ID}")
    expected_scenarios = {scenario["name"] for scenario in config["scenarios"]}
    if set(summary) != expected_scenarios:
        raise ValueError("summary scenarios do not match the frozen protocol")

    scenario_results: dict[str, Any] = {}
    passes = 0
    runtime_passes = 0
    for scenario in sorted(expected_scenarios):
        uniform = summary[scenario][PRIMARY_UNIFORM]
        online = summary[scenario][PRIMARY_ONLINE]
        auc_difference = float(online["success_auc_mean"]) - float(
            uniform["success_auc_mean"]
        )
        uniform_distractor_kl = float(uniform["cumulative_distractor_kl_mean"])
        online_distractor_kl = float(online["cumulative_distractor_kl_mean"])
        distractor_kl_ratio = online_distractor_kl / uniform_distractor_kl
        runtime_ratio = float(online["runtime_seconds_mean"]) / float(
            uniform["runtime_seconds_mean"]
        )
        auc_pass = auc_difference >= -MAX_AUC_DEFICIT
        kl_pass = distractor_kl_ratio <= MAX_DISTRACTOR_KL_RATIO
        runtime_pass = runtime_ratio <= MAX_RUNTIME_RATIO
        scenario_pass = auc_pass and kl_pass
        passes += int(scenario_pass)
        runtime_passes += int(runtime_pass)
        scenario_results[scenario] = {
            "uniform_auc": float(uniform["success_auc_mean"]),
            "online_auc": float(online["success_auc_mean"]),
            "auc_difference": auc_difference,
            "uniform_distractor_kl": uniform_distractor_kl,
            "online_distractor_kl": online_distractor_kl,
            "distractor_kl_ratio": distractor_kl_ratio,
            "runtime_ratio": runtime_ratio,
            "auc_pass": auc_pass,
            "kl_pass": kl_pass,
            "runtime_pass": runtime_pass,
            "scenario_pass": scenario_pass,
        }

    return {
        "protocol_id": PROTOCOL_ID,
        "execution_commit": "d37a5ff",
        "primary_pair": {
            "uniform": PRIMARY_UNIFORM,
            "online": PRIMARY_ONLINE,
        },
        "frozen_thresholds": {
            "max_auc_deficit": MAX_AUC_DEFICIT,
            "max_distractor_kl_ratio": MAX_DISTRACTOR_KL_RATIO,
            "max_runtime_ratio": MAX_RUNTIME_RATIO,
            "required_scenario_passes": REQUIRED_SCENARIO_PASSES,
        },
        "scenario_results": scenario_results,
        "scenario_pass_count": passes,
        "runtime_pass_count": runtime_passes,
        "generalization_decision": (
            "GO" if passes >= REQUIRED_SCENARIO_PASSES else "NO-GO"
        ),
        "systems_feasibility_decision": (
            "PASS" if runtime_passes == len(expected_scenarios) else "FAIL"
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
