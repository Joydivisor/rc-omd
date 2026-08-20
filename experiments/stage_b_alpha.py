"""Continuous `alpha` estimator for `stage-b-mlp-2026-08-19`.

    S_jk   = max(0, <G_j, G_k> / (||G_j|| ||G_k||))
    m_k    = sum_j S_jk
    b_k    = sum_j S_jk * s_j
    alpha  = 1 - (1/H) * sum_k |b_k| / m_k

where ``G_k`` is the flattened Jacobian of position ``k``'s logit vector with
respect to every trainable parameter and ``s_k`` is +1 for critical positions,
-1 for distractors.

Under a softmax-linear head this reduces **exactly** to the discrete aliasing
index, which is the protocol's calibration gate. Under the MLP it does not, and
is not expected to: the output bias contributes an identical derivative at every
position and hidden activations of distinct tie-groups are not orthogonal. See
the calibration amendment in `docs/STAGE_B_MLP_PROTOCOL.md`.

`alpha` is measured on the Uniform baseline arm only. RWP-OMD never enters it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.run_reliability_diagnostics import build_algorithm, build_environment


def position_jacobians(algorithm: Any) -> np.ndarray:
    """Per-position Jacobian of logits w.r.t. all trainable parameters."""

    head = getattr(algorithm, "head", None)
    if head is not None:
        return head.position_jacobians(algorithm.features)

    # Softmax-linear head: logits = features @ weights, so
    # d logits_k[a] / d weights[j, a'] = delta(a, a') * features[k, j].
    features = np.asarray(algorithm.features, dtype=np.float64)
    horizon, dimension = features.shape
    n_actions = int(algorithm.n_actions)
    rows = np.zeros((horizon, n_actions * dimension * n_actions))
    for k in range(horizon):
        for action in range(n_actions):
            block = np.zeros((dimension, n_actions))
            block[:, action] = features[k]
            start = action * dimension * n_actions
            rows[k, start:start + dimension * n_actions] = block.ravel()
    return rows


def alpha_from_jacobians(
    jacobians: np.ndarray, signs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """The frozen continuous aliasing index."""

    norms = np.linalg.norm(jacobians, axis=1, keepdims=True)
    if not np.all(norms > epsilon):
        # A position with no gradient has no tie-group membership to speak of;
        # the protocol's clipping convention makes S row-degenerate, so the
        # measurement is undefined rather than zero.
        return float("nan")
    unit = jacobians / norms
    similarity = np.clip(unit @ unit.T, 0.0, None)
    m = similarity.sum(axis=0)
    b = similarity @ signs
    return float(1.0 - np.sum(np.abs(b) / m) / jacobians.shape[0])


def measure_alpha(
    scenario: dict[str, Any],
    method_config: dict[str, Any],
    seed: int,
    evaluation_interval: int,
) -> dict[str, Any]:
    """Run the baseline arm and record `alpha` at every checkpoint.

    Mirrors ``run_reliability_diagnostics.run_one``'s loop so the trajectory is
    identical to the one the sweep produces for this arm and seed.
    """

    environment = build_environment(scenario)
    algorithm = build_algorithm("baseline", method_config, scenario, seed)
    rng = np.random.default_rng(seed)
    iterations = int(scenario["iterations"])
    group_size = int(scenario["group_size"])

    critical = set(int(k) for k in scenario["critical_positions"])
    horizon = int(scenario["horizon"])
    signs = np.array(
        [1.0 if k in critical else -1.0 for k in range(horizon)], dtype=np.float64
    )

    checkpoints: list[float] = []
    for iteration in range(iterations + 1):
        if iteration % evaluation_interval == 0:
            checkpoints.append(
                alpha_from_jacobians(position_jacobians(algorithm), signs)
            )
        if iteration == iterations:
            break
        trajectories = environment.sample(algorithm.policy, group_size, rng)
        algorithm.update(trajectories, environment.batch_rewards(trajectories))

    values = np.asarray(checkpoints, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "alpha_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "alpha_initial": float(values[0]),
        "alpha_final": float(values[-1]),
        "n_checkpoints": int(values.size),
        "n_undefined": int(values.size - finite.size),
    }


def measure_alpha_over_seeds(
    scenario: dict[str, Any],
    method_config: dict[str, Any],
    seeds: list[int],
    evaluation_interval: int,
) -> dict[str, Any]:
    """Checkpoint-averaged `alpha`, then averaged over the baseline arm's seeds."""

    per_seed = [
        measure_alpha(scenario, method_config, int(seed), evaluation_interval)
        for seed in seeds
    ]
    means = np.array([entry["alpha_mean"] for entry in per_seed], dtype=np.float64)
    return {
        "alpha": float(np.mean(means)),
        "alpha_sd_over_seeds": float(np.std(means, ddof=1)) if means.size > 1 else 0.0,
        "alpha_initial": float(
            np.mean([entry["alpha_initial"] for entry in per_seed])
        ),
        "alpha_final": float(np.mean([entry["alpha_final"] for entry in per_seed])),
        "seeds": [int(seed) for seed in seeds],
        "per_seed_alpha": [float(value) for value in means],
        "undefined_checkpoints": int(
            sum(entry["n_undefined"] for entry in per_seed)
        ),
    }
