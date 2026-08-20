"""Closed estimator menu for `stage-b2-estimator-2026-08-19`.

Each entry is a fixed deterministic transform of the per-position logit
Jacobian with no free parameters, applied before the cosine Gram matrix and the
unchanged Stage B `alpha` formula. Nothing may be added to this menu after
results are seen.

Entries `action_difference` and `fisher` encode the suspicion that Stage B
repeated the error class of ERRATA E4 -- measuring logit geometry where the
update is governed by parameter movement under the KL metric.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from experiments.run_reliability_diagnostics import build_algorithm, build_environment
from experiments.stage_b_alpha import position_jacobians

PROTOCOL_ID = "stage-b2-estimator-2026-08-19"

# The Stage B estimator, kept as the refuted control per the geometry-v3
# principle: if selection ranks it first, the procedure is overfitting.
REFUTED_ESTIMATOR = "jacobian_cosine"


def _reshape(jacobians: np.ndarray, n_actions: int) -> np.ndarray:
    """(H, A*P) -> (H, A, P)."""

    horizon, width = jacobians.shape
    return jacobians.reshape(horizon, n_actions, width // n_actions)


def _flatten(blocks: np.ndarray) -> np.ndarray:
    return blocks.reshape(blocks.shape[0], -1)


def t_identity(jacobians, context):
    return jacobians


def t_centered(jacobians, context):
    """Remove the across-position mean.

    The output bias contributes an identical derivative at every position, so it
    inflates every similarity uniformly; centring removes that component.
    """

    return jacobians - jacobians.mean(axis=0, keepdims=True)


def t_no_bias(jacobians, context):
    """Drop the bias blocks from every action's Jacobian slice."""

    head = context["head"]
    if head is None:
        return jacobians  # linear head has no bias parameters
    blocks = _reshape(jacobians, context["n_actions"])
    sizes = [head.W1.size, head.b1.size, head.W2.size, head.b2.size]
    offsets = np.cumsum([0] + sizes)
    keep = np.concatenate([
        np.arange(offsets[0], offsets[1]),   # W1
        np.arange(offsets[2], offsets[3]),   # W2
    ])
    return _flatten(blocks[:, :, keep])


def t_action_difference(jacobians, context):
    """Centre across actions: the softmax is invariant to a constant shift."""

    blocks = _reshape(jacobians, context["n_actions"])
    return _flatten(blocks - blocks.mean(axis=1, keepdims=True))


def t_fisher(jacobians, context):
    """Weight by the softmax metric ``diag(pi) - pi pi^T``.

    This replaces Euclidean logit geometry with the metric that governs KL
    movement, which is the quantity `alpha` is about.
    """

    blocks = _reshape(jacobians, context["n_actions"])
    policy = context["policy"]
    out = np.empty_like(blocks)
    for k in range(blocks.shape[0]):
        p = policy[k]
        fisher = np.diag(p) - np.outer(p, p)
        out[k] = fisher @ blocks[k]
    return _flatten(out)


def t_centered_action_difference(jacobians, context):
    return t_centered(t_action_difference(jacobians, context), context)


ESTIMATORS: "OrderedDict[str, Callable]" = OrderedDict([
    ("jacobian_cosine", t_identity),
    ("centered", t_centered),
    ("no_bias", t_no_bias),
    ("action_difference", t_action_difference),
    ("fisher", t_fisher),
    ("centered_action_difference", t_centered_action_difference),
])


def alpha_from_jacobians(
    jacobians: np.ndarray, signs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """Unchanged Stage B formula, applied to a transformed Jacobian."""

    norms = np.linalg.norm(jacobians, axis=1, keepdims=True)
    if not np.all(norms > epsilon):
        return float("nan")
    unit = jacobians / norms
    similarity = np.clip(unit @ unit.T, 0.0, None)
    m = similarity.sum(axis=0)
    b = similarity @ signs
    return float(1.0 - np.sum(np.abs(b) / m) / jacobians.shape[0])


def measure_all(
    scenario: dict[str, Any],
    method_config: dict[str, Any],
    seed: int,
    evaluation_interval: int,
) -> dict[str, float]:
    """Checkpoint-averaged `alpha` for every menu entry, in one baseline run."""

    environment = build_environment(scenario)
    algorithm = build_algorithm("baseline", method_config, scenario, seed)
    rng = np.random.default_rng(seed)
    iterations = int(scenario["iterations"])
    group_size = int(scenario["group_size"])
    critical = {int(k) for k in scenario["critical_positions"]}
    horizon = int(scenario["horizon"])
    signs = np.array(
        [1.0 if k in critical else -1.0 for k in range(horizon)], dtype=np.float64
    )

    running: dict[str, list[float]] = {name: [] for name in ESTIMATORS}
    for iteration in range(iterations + 1):
        if iteration % evaluation_interval == 0:
            raw = position_jacobians(algorithm)
            context = {
                "head": getattr(algorithm, "head", None),
                "n_actions": int(algorithm.n_actions),
                "policy": algorithm.policy,
            }
            for name, transform in ESTIMATORS.items():
                running[name].append(
                    alpha_from_jacobians(transform(raw, context), signs)
                )
        if iteration == iterations:
            break
        trajectories = environment.sample(algorithm.policy, group_size, rng)
        algorithm.update(trajectories, environment.batch_rewards(trajectories))

    out: dict[str, float] = {}
    for name, values in running.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        out[name] = float(np.mean(finite)) if finite.size else float("nan")
    return out


def measure_all_over_seeds(
    scenario: dict[str, Any],
    method_config: dict[str, Any],
    seeds: list[int],
    evaluation_interval: int,
) -> dict[str, float]:
    per_seed = [
        measure_all(scenario, method_config, int(seed), evaluation_interval)
        for seed in seeds
    ]
    return {
        name: float(np.mean([entry[name] for entry in per_seed]))
        for name in ESTIMATORS
    }
