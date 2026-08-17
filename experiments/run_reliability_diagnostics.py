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
    OnlineReliabilityOMD,
    OracleCreditOMD,
    ProjectedGroupOMD,
    ProjectedOnlineReliabilityOMD,
    ReliabilityCalibratedOMD,
    ReliabilityWeightedProjectionOMD,
)
from environments import ControlledSequenceMDP, StructuredSequenceMDP


METHOD_LABELS = {
    "uniform_group_omd": "Uniform Group OMD",
    "entropy_weighted_omd": "Entropy-weighted OMD",
    "global_reliability_omd": "Global-reliability OMD",
    "rc_omd": "Local RC-OMD",
    "oracle_credit_omd": "Oracle-credit OMD",
    "online_rc_omd": "Online RC-OMD",
    "projected_group_omd": "Projected Group OMD",
    "projected_online_rc_omd": "Projected Online RC-OMD",
    "rwp_omd": "RWP-OMD",
}

METHOD_COLORS = {
    "uniform_group_omd": "#2458A6",
    "entropy_weighted_omd": "#D97706",
    "global_reliability_omd": "#7C3AED",
    "rc_omd": "#DC2626",
    "oracle_credit_omd": "#15803D",
    "online_rc_omd": "#0891B2",
    "projected_group_omd": "#2458A6",
    "projected_online_rc_omd": "#0891B2",
    "rwp_omd": "#DC2626",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_environment(scenario: dict[str, Any]) -> Any:
    if scenario.get("environment", "controlled_all") == "threshold":
        return StructuredSequenceMDP.from_sequences(
            horizon=int(scenario["horizon"]),
            n_actions=int(scenario["n_actions"]),
            critical_positions=scenario["critical_positions"],
            target_actions=scenario["target_actions"],
            minimum_matches=int(scenario["minimum_matches"]),
        )
    return ControlledSequenceMDP.from_sequences(
        horizon=int(scenario["horizon"]),
        n_actions=int(scenario["n_actions"]),
        critical_positions=scenario["critical_positions"],
        target_actions=scenario["target_actions"],
    )


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
    if algorithm_name in {
        "projected_group_omd",
        "projected_online_rc_omd",
        "rwp_omd",
    }:
        arguments.update(
            {
                "features": np.asarray(scenario["features"], dtype=np.float64),
                "projection_steps": int(method_config["projection_steps"]),
                "projection_learning_rate": float(
                    method_config["projection_learning_rate"]
                ),
                "projection_ridge": float(method_config["projection_ridge"]),
                "projection_tolerance": float(
                    method_config.get("projection_tolerance", 1e-9)
                ),
            }
        )
        if algorithm_name == "projected_group_omd":
            return ProjectedGroupOMD(**arguments)
        arguments.update(
            {
                "reliability_decay": float(method_config["reliability_decay"]),
                "confidence_multiplier": float(method_config["confidence_multiplier"]),
                "warmup_effective_samples": float(
                    method_config["warmup_effective_samples"]
                ),
                "reliability_floor": float(method_config["reliability_floor"]),
            }
        )
        if algorithm_name == "rwp_omd":
            arguments["projection_lambda"] = float(
                method_config["projection_lambda"]
            )
            return ReliabilityWeightedProjectionOMD(**arguments)
        return ProjectedOnlineReliabilityOMD(**arguments)

    if algorithm_name in {"global_reliability_omd", "rc_omd"}:
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
        return ReliabilityCalibratedOMD(**arguments)
    if algorithm_name == "online_rc_omd":
        arguments.update(
            {
                "reliability_decay": float(method_config["reliability_decay"]),
                "confidence_multiplier": float(method_config["confidence_multiplier"]),
                "warmup_effective_samples": float(
                    method_config["warmup_effective_samples"]
                ),
                "reliability_floor": float(method_config["reliability_floor"]),
            }
        )
        return OnlineReliabilityOMD(**arguments)
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
    estimate = getattr(algorithm, "last_credit_estimate", None)
    last_reward_std = getattr(algorithm, "last_reward_std", None)
    if estimate is None:
        return {
            "critical_reliability": float("nan"),
            "distractor_reliability": float("nan"),
            "reliability_topk_precision": float("nan"),
        }
    if last_reward_std is None or last_reward_std <= 1e-12:
        return {
            "critical_reliability": float("nan"),
            "distractor_reliability": float("nan"),
            "reliability_topk_precision": float("nan"),
        }
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
    environment = build_environment(scenario)
    algorithm = build_algorithm(method, method_config, scenario, seed)
    rng = np.random.default_rng(seed)
    iterations = int(scenario["iterations"])
    group_size = int(scenario["group_size"])
    critical = np.asarray(environment.critical_positions, dtype=np.int64)
    distractors = np.asarray(environment.distractor_positions, dtype=np.int64)

    history: list[dict[str, Any]] = []
    cumulative_critical_kl = 0.0
    cumulative_distractor_kl = 0.0
    cumulative_projection_kl = 0.0
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
                    "cumulative_projection_kl": cumulative_projection_kl,
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
        cumulative_projection_kl += float(stats.get("projection_kl", 0.0))

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
            critical_kl: list[float] = []
            distractor_kl: list[float] = []
            harmful_rate: list[float] = []
            runtime: list[float] = []
            critical_reliability: list[float] = []
            distractor_reliability: list[float] = []
            topk_precision: list[float] = []
            projection_kl: list[float] = []
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
                critical_kl.append(float(seed_rows[-1]["cumulative_critical_kl"]))
                distractor_kl.append(float(seed_rows[-1]["cumulative_distractor_kl"]))
                harmful_rate.append(float(seed_rows[-1]["harmful_update_rate"]))
                runtime.append(run_metadata[(scenario, method, seed)]["runtime_seconds"])
                projection_kl.append(
                    float(seed_rows[-1].get("cumulative_projection_kl", 0.0))
                )
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
                "success_auc_std": float(np.std(auc)),
                "distractor_kl_fraction_mean": float(np.mean(distractor_fraction)),
                "cumulative_critical_kl_mean": float(np.mean(critical_kl)),
                "cumulative_distractor_kl_mean": float(np.mean(distractor_kl)),
                "cumulative_distractor_kl_std": float(np.std(distractor_kl)),
                "cumulative_total_kl_mean": float(
                    np.mean(np.asarray(critical_kl) + np.asarray(distractor_kl))
                ),
                "harmful_update_rate_mean": float(np.mean(harmful_rate)),
                "runtime_seconds_mean": float(np.mean(runtime)),
                "cumulative_projection_kl_mean": float(np.mean(projection_kl)),
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
                # Per-seed values, ordered by the sorted `seeds` list above. Every
                # method within a scenario is run against the same seed list (see
                # `main`), and `run_one` seeds its environment RNG from `seed`
                # alone (no method or step-size component), so index i in every
                # per-seed array below refers to the same seed across methods.
                # This is what makes paired, same-seed comparisons (e.g. the
                # Pareto V1 protocol) valid without re-deriving pairing from
                # `history.csv`.
                "seeds": [int(seed) for seed in seeds],
                "success_auc_per_seed": auc,
                "cumulative_distractor_kl_per_seed": distractor_kl,
                "cumulative_critical_kl_per_seed": critical_kl,
                "runtime_seconds_per_seed": runtime,
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


def plot_pareto(summary: dict[str, Any], output_stem: Path) -> None:
    """Plot mean AUC against absolute distractor KL with across-seed SD."""

    scenarios = list(summary)
    all_values = [value for methods in summary.values() for value in methods.values()]
    max_x = max(float(value["cumulative_distractor_kl_mean"]) for value in all_values)
    min_y = min(float(value["success_auc_mean"]) for value in all_values)
    max_y = max(float(value["success_auc_mean"]) for value in all_values)
    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(7 * len(scenarios), 5),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios, strict=True):
        for index, (method, values) in enumerate(summary[scenario].items()):
            if method.startswith("online"):
                marker, color = "o", "#0891B2"
            elif method.startswith("uniform"):
                marker, color = "s", "#2458A6"
            elif "bootstrap" in method or method.startswith("rc_"):
                marker, color = "^", "#DC2626"
            elif "oracle" in method:
                marker, color = "*", "#15803D"
            else:
                marker, color = "D", "#7C3AED"
            label = METHOD_LABELS.get(method, method.replace("_", " ").title())
            axis.errorbar(
                float(values["cumulative_distractor_kl_mean"]),
                float(values["success_auc_mean"]),
                xerr=float(values["cumulative_distractor_kl_std"]),
                yerr=float(values["success_auc_std"]),
                marker=marker,
                markersize=6,
                linestyle="none",
                color=color,
                capsize=2,
                label=label,
                alpha=0.9,
            )
            short_label = label
            if short_label.startswith("Online Eta"):
                value = float(short_label.removeprefix("Online Eta")) / 100.0
                short_label = f"Online eta={value:.2f}"
            elif short_label.startswith("Uniform Eta"):
                value = float(short_label.removeprefix("Uniform Eta")) / 100.0
                short_label = f"Uniform eta={value:.2f}"
            elif short_label.startswith("Online Decay"):
                value = float(short_label.removeprefix("Online Decay")) / 100.0
                short_label = f"decay={value:.2f}"
            axis.annotate(
                short_label,
                (
                    float(values["cumulative_distractor_kl_mean"]),
                    float(values["success_auc_mean"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color=color,
            )
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_xlabel("Cumulative KL on distractor positions")
        axis.set_xlim(left=0.0, right=max_x * 1.08 if max_x > 0.0 else 1.0)
        axis.set_ylim(min_y - 0.01, min(1.0, max_y + 0.01))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized success AUC")
    axes[-1].legend(fontsize=7, loc="lower right")
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
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
    plot_pareto(summary, output_directory / "auc_distractor_kl_pareto")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
