"""Run the Uniform Group OMD baseline on a controlled sequence task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from algorithms import GroupOMD
from environments import ControlledSequenceMDP


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_seed(config: dict[str, Any], seed: int) -> list[dict[str, float]]:
    environment = ControlledSequenceMDP.from_sequences(
        horizon=config["environment"]["horizon"],
        n_actions=config["environment"]["n_actions"],
        critical_positions=config["environment"]["critical_positions"],
        target_actions=config["environment"]["target_actions"],
    )
    algorithm = GroupOMD(
        horizon=environment.horizon,
        n_actions=environment.n_actions,
        step_size=config["algorithm"]["step_size"],
        normalize_advantages=config["algorithm"].get("normalize_advantages", False),
        importance_clip=config["algorithm"].get("importance_clip", 20.0),
    )
    rng = np.random.default_rng(seed)
    group_size = int(config["training"]["group_size"])
    iterations = int(config["training"]["iterations"])
    evaluation_interval = int(config["training"].get("evaluation_interval", 1))

    history: list[dict[str, float]] = []
    zero_variance_count = 0.0
    for iteration in range(iterations + 1):
        if iteration % evaluation_interval == 0:
            policy = algorithm.policy
            target_probabilities = [
                policy[position, target]
                for position, target in zip(
                    environment.critical_positions,
                    environment.target_actions,
                    strict=True,
                )
            ]
            history.append(
                {
                    "seed": float(seed),
                    "iteration": float(iteration),
                    "success_probability": environment.expected_success_probability(policy),
                    "mean_target_probability": float(np.mean(target_probabilities)),
                    "policy_entropy": algorithm.entropy(),
                    "zero_variance_ratio": zero_variance_count / max(iteration, 1),
                }
            )
        if iteration == iterations:
            break

        trajectories = environment.sample(algorithm.policy, group_size, rng)
        rewards = environment.batch_rewards(trajectories)
        update_stats = algorithm.update(trajectories, rewards)
        zero_variance_count += update_stats["zero_variance_group"]

    return history


def write_history(path: Path, histories: list[list[dict[str, float]]]) -> None:
    rows = [row for history in histories for row in history]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, histories: list[list[dict[str, float]]]) -> None:
    final_success = np.asarray(
        [history[-1]["success_probability"] for history in histories],
        dtype=np.float64,
    )
    summary = {
        "n_seeds": len(histories),
        "final_success_mean": float(final_success.mean()),
        "final_success_std": float(final_success.std()),
        "final_success_min": float(final_success.min()),
        "final_success_max": float(final_success.max()),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def plot_histories(path: Path, histories: list[list[dict[str, float]]]) -> None:
    iterations = np.asarray([row["iteration"] for row in histories[0]])
    success = np.asarray(
        [[row["success_probability"] for row in history] for history in histories]
    )
    mean = success.mean(axis=0)
    std = success.std(axis=0)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(iterations, mean, label="Uniform Group OMD", color="#2458A6")
    axis.fill_between(iterations, mean - std, mean + std, color="#2458A6", alpha=0.2)
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Exact success probability")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequence_baseline.json"),
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    histories = [train_seed(config, int(seed)) for seed in config["training"]["seeds"]]

    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    write_history(output_directory / "history.csv", histories)
    write_summary(output_directory / "summary.json", histories)
    plot_histories(output_directory / "success_probability.png", histories)

    final_success = [history[-1]["success_probability"] for history in histories]
    print(
        "Uniform Group OMD final exact success probability: "
        f"{np.mean(final_success):.4f} +/- {np.std(final_success):.4f}"
    )
    print(f"Results written to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
