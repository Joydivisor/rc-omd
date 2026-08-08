"""Shared inverse-propensity action-score estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def inverse_propensity_action_scores(
    trajectories: IntArray,
    step_signals: FloatArray,
    policy: FloatArray,
    *,
    importance_clip: float,
    min_probability: float,
) -> FloatArray:
    """Estimate linearized per-position action gains from sampled trajectories."""

    batch = np.asarray(trajectories, dtype=np.int64)
    signals = np.asarray(step_signals, dtype=np.float64)
    probabilities = np.asarray(policy, dtype=np.float64)
    if batch.ndim != 2:
        raise ValueError("trajectories must be a two-dimensional array")
    group_size, horizon = batch.shape
    if signals.shape == (group_size,):
        signals = np.repeat(signals[:, None], horizon, axis=1)
    if signals.shape != batch.shape:
        raise ValueError("step_signals must match trajectories or contain one per row")
    if probabilities.ndim != 2 or probabilities.shape[0] != horizon:
        raise ValueError("policy horizon must match trajectories")
    n_actions = probabilities.shape[1]
    if np.any(batch < 0) or np.any(batch >= n_actions):
        raise ValueError("trajectory contains an action outside the policy")
    if not np.all(np.isfinite(signals)):
        raise ValueError("step_signals must be finite")

    scores = np.zeros((horizon, n_actions), dtype=np.float64)
    for position in range(horizon):
        for action in range(n_actions):
            selected = batch[:, position] == action
            if not np.any(selected):
                continue
            inverse_propensity = min(
                1.0 / max(float(probabilities[position, action]), min_probability),
                importance_clip,
            )
            scores[position, action] = (
                float(signals[selected, position].sum())
                * inverse_propensity
                / group_size
            )
    return scores
