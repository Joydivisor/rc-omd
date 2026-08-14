"""Compare uniform, entropy-weighted, and oracle-credit OMD baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from algorithms import EntropyWeightedOMD, GroupOMD, OracleCreditOMD
from algorithms.credit_weighted_omd import policy_entropy_by_position
from environments import ControlledSequenceMDP


METHOD_LABELS = {
    "uniform_group_omd": "Uniform Group OMD",
    "entropy_weighted_omd": "Entropy-weighted OMD",
    "oracle_credit_omd": "Oracle-credit OMD",
}

METHOD_COLORS = {
    "uniform_group_omd": "#2458A6",
    "entropy_weighted_omd": "#D97706",
    "oracle_credit_omd": "#15803D",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_algorithm(
    method: str,
    method_config: dict[str, Any],
    scenario: dict[str, Any],
) -> GroupOMD:
    arguments = {
        "horizon": int(scenario["horizon"]),
        "n_actions": int(scenario["n_actions"]),
        "step_size": float(method_config["step_size"]),
        "normalize_advantages": bool(method_config.get("normalize_advantages", False)),
        "importance_clip": float(method_config.get("importance_clip", 20.0)),
        "initial_policy": np.asarray(scenario["initial_policy"], dtype=np.float64),
    }
    if method == "uniform_group_omd":
        return GroupOMD(**arguments)
    if method == "entropy_weighted_omd":
        return EntropyWeightedOMD(**arguments)
    if method == "oracle_credit_omd":
        return OracleCreditOMD(**arguments)
    raise ValueError(f"unknown method: {method}")


def proxy_diagnostics(
    environment: ControlledSequenceMDP,
    policy: np.ndarray,
) -> dict[str, float]:
    entropy = policy_entropy_by_position(policy)
    importance = environment.oracle_position_importance(policy)

    entropy_distribution = entropy / max(float(entropy.sum()), 1e-12)
    importance_distribution = importance / max(float(importance.sum()), 1e-12)
    proxy_mse = float(np.mean((entropy_distribution - importance_distribution) ** 2))

    n_critical = len(environment.critical_positions)
    top_entropy = set(np.argsort(entropy)[-n_critical:].tolist())
    critical = set(environment.critical_positions)
    top_precision = len(top_entropy & critical) / n_critical

    if float(entropy.std()) <= 1e-12 or float(importance.std()) <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(entropy, importance)[0, 1])

    return {
        "entropy_credit_correlation": correlation,
        "entropy_credit_mse": proxy_mse,
        "entropy_topk_precision": float(top_precision),
    }


def train_run(
    config: dict[str, Any],
    scenario_index: int,
    method_index: int,
    seed: int,
) -> list[dict[str, float | str]]:
    scenario = config["scenarios"][scenario_index]
    method = list(config["methods"])[method_index]
    environment = ControlledSequenceMDP.from_sequences(
        horizon=scenario["horizon"],
        n_actions=scenario["n_actions"],
        critical_positions=scenario["critical_positions"],
        target_actions=scenario["target_actions"],
    )
    algorithm = build_algorithm(method, config["methods"][method], scenario)
    # Common random numbers: the stream depends on the seed and scenario only.
    # Including method_index gave each method a different trajectory stream at
    # the same nominal seed, making the comparison unpaired and dependent on the
    # order methods happen to appear in the config.
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_index]))

    iterations = int(config["training"]["iterations"])
    group_size = int(config["training"]["group_size"])
    evaluation_interval = int(config["training"]["evaluation_interval"])
    zero_variance_count = 0.0
    cumulative_kl = 0.0
    history: list[dict[str, float | str]] = []

    for iteration in range(iterations + 1):
        if iteration % evaluation_interval == 0:
            policy = algorithm.policy
            diagnostics = proxy_diagnostics(environment, policy)
            history.append(
                {
                    "scenario": scenario["name"],
                    "method": method,
                    "seed": float(seed),
                    "iteration": float(iteration),
                    "success_probability": environment.expected_success_probability(policy),
                    "policy_entropy": algorithm.entropy(),
                    "zero_variance_ratio": zero_variance_count / max(iteration, 1),
                    "cumulative_kl": cumulative_kl,
                    **diagnostics,
                }
            )
        if iteration == iterations:
            break

        trajectories = environment.sample(algorithm.policy, group_size, rng)
        rewards = environment.batch_rewards(trajectories)
        if isinstance(algorithm, OracleCreditOMD):
            credits = environment.oracle_batch_credit(trajectories, algorithm.policy)
            stats = algorithm.update_with_credit(trajectories, rewards, credits)
        else:
            stats = algorithm.update(trajectories, rewards)
        zero_variance_count += stats["zero_variance_group"]
        cumulative_kl += stats["kl_drift"]

    return history


def summarize(
    config: dict[str, Any],
    histories: list[list[dict[str, float | str]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        scenario_name = scenario["name"]
        summary[scenario_name] = {}
        for method in config["methods"]:
            selected = [
                history
                for history in histories
                if history[0]["scenario"] == scenario_name and history[0]["method"] == method
            ]
            final_success = np.asarray(
                [float(history[-1]["success_probability"]) for history in selected]
            )
            auc = np.asarray(
                [
                    float(
                        np.trapezoid(
                            [float(row["success_probability"]) for row in history],
                            [float(row["iteration"]) for row in history],
                        )
                        / float(history[-1]["iteration"])
                    )
                    for history in selected
                ]
            )
            summary[scenario_name][method] = {
                "final_success_mean": float(final_success.mean()),
                "final_success_std": float(final_success.std()),
                "success_auc_mean": float(auc.mean()),
                "success_auc_std": float(auc.std()),
                "initial_entropy_topk_precision": float(
                    selected[0][0]["entropy_topk_precision"]
                ),
                "initial_entropy_credit_correlation": float(
                    selected[0][0]["entropy_credit_correlation"]
                ),
            }
    return summary


def write_history(path: Path, histories: list[list[dict[str, float | str]]]) -> None:
    rows = [row for history in histories for row in history]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_success(
    path: Path,
    config: dict[str, Any],
    histories: list[list[dict[str, float | str]]],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(config["scenarios"]),
        figsize=(6.4 * len(config["scenarios"]), 4.3),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, scenario in zip(axes, config["scenarios"], strict=True):
        for method in config["methods"]:
            selected = [
                history
                for history in histories
                if history[0]["scenario"] == scenario["name"]
                and history[0]["method"] == method
            ]
            iterations = np.asarray([float(row["iteration"]) for row in selected[0]])
            success = np.asarray(
                [
                    [float(row["success_probability"]) for row in history]
                    for history in selected
                ]
            )
            mean = success.mean(axis=0)
            std = success.std(axis=0)
            axis.plot(
                iterations,
                mean,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
            axis.fill_between(
                iterations,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=METHOD_COLORS[method],
                alpha=0.16,
            )
        axis.set_title(scenario["name"].replace("_", " ").title())
        axis.set_xlabel("Training iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Exact success probability")
    axes[-1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_initial_diagnostics(path: Path, config: dict[str, Any]) -> None:
    figure, axes = plt.subplots(
        1,
        len(config["scenarios"]),
        figsize=(6.4 * len(config["scenarios"]), 4.3),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    for axis, scenario in zip(axes, config["scenarios"], strict=True):
        environment = ControlledSequenceMDP.from_sequences(
            horizon=scenario["horizon"],
            n_actions=scenario["n_actions"],
            critical_positions=scenario["critical_positions"],
            target_actions=scenario["target_actions"],
        )
        policy = np.asarray(scenario["initial_policy"], dtype=np.float64)
        entropy = policy_entropy_by_position(policy)
        importance = environment.oracle_position_importance(policy)
        entropy /= max(float(entropy.max()), 1e-12)
        importance /= max(float(importance.max()), 1e-12)

        positions = np.arange(environment.horizon)
        width = 0.38
        axis.bar(positions - width / 2, entropy, width, label="Normalized entropy")
        axis.bar(
            positions + width / 2,
            importance,
            width,
            label="Normalized oracle importance",
        )
        axis.set_title(scenario["name"].replace("_", " ").title())
        axis.set_xlabel("Sequence position")
        axis.set_xticks(positions)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized score")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/credit_diagnostics.json"),
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)

    histories = [
        train_run(config, scenario_index, method_index, int(seed))
        for scenario_index in range(len(config["scenarios"]))
        for method_index in range(len(config["methods"]))
        for seed in config["training"]["seeds"]
    ]

    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    write_history(output_directory / "history.csv", histories)
    summary = summarize(config, histories)
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_success(output_directory / "success_comparison.png", config, histories)
    plot_initial_diagnostics(output_directory / "initial_credit_diagnostics.png", config)

    print(json.dumps(summary, indent=2))
    print(f"Results written to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
