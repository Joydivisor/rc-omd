"""Evaluate local reliability calibration against global and credit baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from algorithms import (
    EntropyWeightedOMD,
    GlobalReliabilityOMD,
    GroupOMD,
    OracleCreditOMD,
    ReliabilityCalibratedOMD,
)
from environments import ControlledSequenceMDP


METHOD_LABELS = {
    "uniform_group_omd": "Uniform Group OMD",
    "entropy_weighted_omd": "Entropy-weighted OMD",
    "global_reliability_omd": "Global-reliability OMD",
    "rc_omd": "Local RC-OMD",
    "oracle_credit_omd": "Oracle-credit OMD",
}

METHOD_COLORS = {
    "uniform_group_omd": "#2458A6",
    "entropy_weighted_omd": "#D97706",
    "global_reliability_omd": "#7C3AED",
    "rc_omd": "#DC2626",
    "oracle_credit_omd": "#15803D",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_algorithm(
    method: str,
    method_config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
) -> GroupOMD:
    algorithm_name = str(method_config.get("algorithm", method))
    arguments: dict[str, Any] = {
        "horizon": int(scenario["horizon"]),
        "n_actions": int(scenario["n_actions"]),
        "step_size": float(method_config["step_size"]),
        "normalize_advantages": False,
        "importance_clip": 20.0,
        "initial_policy": np.asarray(scenario["initial_policy"], dtype=np.float64),
    }
    if algorithm_name == "uniform_group_omd":
        return GroupOMD(**arguments)
    if algorithm_name == "entropy_weighted_omd":
        return EntropyWeightedOMD(**arguments)
    if algorithm_name == "oracle_credit_omd":
        return OracleCreditOMD(**arguments)

    arguments.update(
        {
            "bootstrap_samples": int(method_config["bootstrap_samples"]),
            "confidence_multiplier": float(method_config["confidence_multiplier"]),
            "reliability_floor": float(method_config["reliability_floor"]),
            "estimator_seed": seed + 100_000,
        }
    )
    if algorithm_name == "global_reliability_omd":
        return GlobalReliabilityOMD(**arguments)
    if algorithm_name == "rc_omd":
        return ReliabilityCalibratedOMD(**arguments)
    raise ValueError(f"unknown algorithm: {algorithm_name}")


def row_kl(new_policy: np.ndarray, old_policy: np.ndarray) -> np.ndarray:
    floor = 1e-300
    return np.sum(
        new_policy
        * (
            np.log(np.maximum(new_policy, floor))
            - np.log(np.maximum(old_policy, floor))
        ),
        axis=1,
    )


def reliability_diagnostics(
    algorithm: GroupOMD,
    environment: ControlledSequenceMDP,
) -> dict[str, float]:
    if not isinstance(algorithm, ReliabilityCalibratedOMD):
        return {
            "critical_reliability": float("nan"),
            "distractor_reliability": float("nan"),
            "reliability_topk_precision": float("nan"),
        }
    if algorithm.last_reward_std is None or algorithm.last_reward_std <= 1e-12:
        return {
            "critical_reliability": float("nan"),
            "distractor_reliability": float("nan"),
            "reliability_topk_precision": float("nan"),
        }
    estimate = algorithm.last_credit_estimate
    if estimate is None:
        raise RuntimeError("reliability diagnostics requested before an update")
    critical = np.asarray(environment.critical_positions, dtype=np.int64)
    distractors = np.asarray(environment.distractor_positions, dtype=np.int64)
    top_positions = set(
        np.argsort(estimate.reliability)[-len(environment.critical_positions) :].tolist()
    )
    precision = len(top_positions & set(environment.critical_positions)) / len(critical)
    return {
        "critical_reliability": float(estimate.reliability[critical].mean()),
        "distractor_reliability": float(estimate.reliability[distractors].mean()),
        "reliability_topk_precision": float(precision),
    }


def run_one(
    scenario: dict[str, Any],
    method: str,
    method_config: dict[str, Any],
    seed: int,
    evaluation_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    environment = ControlledSequenceMDP.from_sequences(
        horizon=int(scenario["horizon"]),
        n_actions=int(scenario["n_actions"]),
        critical_positions=scenario["critical_positions"],
        target_actions=scenario["target_actions"],
    )
    algorithm = build_algorithm(method, method_config, scenario, seed)
    rng = np.random.default_rng(seed)
    iterations = int(scenario["iterations"])
    group_size = int(scenario["group_size"])
    critical = np.asarray(environment.critical_positions, dtype=np.int64)
    distractors = np.asarray(environment.distractor_positions, dtype=np.int64)

    history: list[dict[str, Any]] = []
    cumulative_critical_kl = 0.0
    cumulative_distractor_kl = 0.0
    harmful_updates = 0
    zero_variance_groups = 0
    start = time.perf_counter()

    for iteration in range(iterations + 1):
        success_before = environment.expected_success_probability(algorithm.policy)
        if iteration % evaluation_interval == 0:
            total_kl = cumulative_critical_kl + cumulative_distractor_kl
            history.append(
                {
                    "scenario": scenario["name"],
                    "method": method,
                    "seed": seed,
                    "iteration": iteration,
                    "success_probability": success_before,
                    "cumulative_critical_kl": cumulative_critical_kl,
                    "cumulative_distractor_kl": cumulative_distractor_kl,
                    "distractor_kl_fraction": (
                        cumulative_distractor_kl / total_kl if total_kl > 0.0 else 0.0
                    ),
                    "harmful_update_rate": harmful_updates / max(iteration, 1),
                    "zero_variance_ratio": zero_variance_groups / max(iteration, 1),
                    **(
                        reliability_diagnostics(algorithm, environment)
                        if iteration > 0
                        else {
                            "critical_reliability": float("nan"),
                            "distractor_reliability": float("nan"),
                            "reliability_topk_precision": float("nan"),
                        }
                    ),
                }
            )
        if iteration == iterations:
            break

        old_policy = algorithm.policy.copy()
        trajectories = environment.sample(old_policy, group_size, rng)
        rewards = environment.batch_rewards(trajectories)
        if isinstance(algorithm, OracleCreditOMD):
            if not isinstance(algorithm, OracleCreditOMD):
                raise TypeError("oracle method must build OracleCreditOMD")
            credits = environment.oracle_batch_credit(trajectories, old_policy)
            stats = algorithm.update_with_credit(trajectories, rewards, credits)
        else:
            stats = algorithm.update(trajectories, rewards)

        position_kl = row_kl(algorithm.policy, old_policy)
        cumulative_critical_kl += float(position_kl[critical].sum())
        cumulative_distractor_kl += float(position_kl[distractors].sum())
        success_after = environment.expected_success_probability(algorithm.policy)
        harmful_updates += int(success_after + 1e-12 < success_before)
        zero_variance_groups += int(stats["zero_variance_group"] > 0.5)

    runtime_seconds = time.perf_counter() - start
    return history, {
        "runtime_seconds": runtime_seconds,
        "harmful_updates": float(harmful_updates),
        "zero_variance_groups": float(zero_variance_groups),
    }


def summarize(
    rows: list[dict[str, Any]],
    run_metadata: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    scenarios = sorted({str(row["scenario"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    for scenario in scenarios:
        summary[scenario] = {}
        for method in methods:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            if not selected:
                continue
            seeds = sorted({int(row["seed"]) for row in selected})
            final_success: list[float] = []
            auc: list[float] = []
            distractor_fraction: list[float] = []
            harmful_rate: list[float] = []
            runtime: list[float] = []
            critical_reliability: list[float] = []
            distractor_reliability: list[float] = []
            topk_precision: list[float] = []
            for seed in seeds:
                seed_rows = sorted(
                    [row for row in selected if int(row["seed"]) == seed],
                    key=lambda row: int(row["iteration"]),
                )
                x = np.asarray([row["iteration"] for row in seed_rows], dtype=float)
                y = np.asarray([row["success_probability"] for row in seed_rows])
                final_success.append(float(y[-1]))
                auc.append(float(np.trapezoid(y, x) / max(float(x[-1]), 1.0)))
                distractor_fraction.append(float(seed_rows[-1]["distractor_kl_fraction"]))
                harmful_rate.append(float(seed_rows[-1]["harmful_update_rate"]))
                runtime.append(run_metadata[(scenario, method, seed)]["runtime_seconds"])
                valid = [
                    row
                    for row in seed_rows
                    if np.isfinite(float(row["critical_reliability"]))
                ]
                if valid:
                    critical_reliability.append(
                        float(np.mean([row["critical_reliability"] for row in valid]))
                    )
                    distractor_reliability.append(
                        float(np.mean([row["distractor_reliability"] for row in valid]))
                    )
                    topk_precision.append(
                        float(np.mean([row["reliability_topk_precision"] for row in valid]))
                    )
            summary[scenario][method] = {
                "final_success_mean": float(np.mean(final_success)),
                "final_success_std": float(np.std(final_success)),
                "success_auc_mean": float(np.mean(auc)),
                "distractor_kl_fraction_mean": float(np.mean(distractor_fraction)),
                "harmful_update_rate_mean": float(np.mean(harmful_rate)),
                "runtime_seconds_mean": float(np.mean(runtime)),
                "critical_reliability_mean": (
                    float(np.mean(critical_reliability))
                    if critical_reliability
                    else None
                ),
                "distractor_reliability_mean": (
                    float(np.mean(distractor_reliability))
                    if distractor_reliability
                    else None
                ),
                "reliability_topk_precision_mean": (
                    float(np.mean(topk_precision)) if topk_precision else None
                ),
            }
    return summary


def plot_success(rows: list[dict[str, Any]], output_path: Path) -> None:
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in rows))
    figure, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios, strict=True):
        methods = list(dict.fromkeys(str(row["method"]) for row in rows))
        for method in methods:
            label = METHOD_LABELS.get(method, method.replace("_", " ").title())
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            if not selected:
                continue
            iterations = sorted({int(row["iteration"]) for row in selected})
            means = [
                np.mean(
                    [
                        float(row["success_probability"])
                        for row in selected
                        if int(row["iteration"]) == iteration
                    ]
                )
                for iteration in iterations
            ]
            axis.plot(
                iterations,
                means,
                label=label,
                color=METHOD_COLORS.get(method),
            )
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_xlabel("Training iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Exact success probability")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_kl_summary(summary: dict[str, Any], output_path: Path) -> None:
    scenarios = list(summary)
    methods = list(next(iter(summary.values())))
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / len(scenarios)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for index, scenario in enumerate(scenarios):
        values = [summary[scenario][method]["distractor_kl_fraction_mean"] for method in methods]
        axis.bar(
            x + (index - (len(scenarios) - 1) / 2) * width,
            values,
            width=width,
            label=scenario.replace("_", " ").title(),
        )
    axis.set_xticks(
        x,
        [METHOD_LABELS.get(method, method.replace("_", " ").title()) for method in methods],
        rotation=18,
    )
    axis.set_ylabel("Fraction of cumulative KL on distractors")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_history(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_metadata: dict[tuple[str, str, int], dict[str, float]] = {}
    evaluation_interval = int(config["training"]["evaluation_interval"])
    for scenario in config["scenarios"]:
        for method, method_config in config["methods"].items():
            for seed in config["training"]["seeds"]:
                run_rows, metadata = run_one(
                    scenario,
                    method,
                    method_config,
                    int(seed),
                    evaluation_interval,
                )
                rows.extend(run_rows)
                run_metadata[(scenario["name"], method, int(seed))] = metadata

    summary = summarize(rows, run_metadata)
    write_history(rows, output_directory / "history.csv")
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_success(rows, output_directory / "success_comparison.png")
    plot_kl_summary(summary, output_directory / "distractor_kl_fraction.png")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
