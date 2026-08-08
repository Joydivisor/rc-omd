"""Uniform group-relative Online Mirror Descent baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class GroupOMD:
    """Exponentiated-gradient OMD with group-relative terminal advantages.

    A trajectory-level advantage is broadcast uniformly across sequence
    positions. Per-action linearized gains are estimated with inverse
    propensity weighting before the KL-mirror update.
    """

    def __init__(
        self,
        *,
        horizon: int,
        n_actions: int,
        step_size: float,
        normalize_advantages: bool = False,
        importance_clip: float = 20.0,
        min_probability: float = 1e-8,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least two")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if importance_clip <= 0:
            raise ValueError("importance_clip must be positive")
        if not 0 < min_probability < 1 / n_actions:
            raise ValueError("min_probability must be in (0, 1 / n_actions)")

        self.horizon = horizon
        self.n_actions = n_actions
        self.step_size = float(step_size)
        self.normalize_advantages = normalize_advantages
        self.importance_clip = float(importance_clip)
        self.min_probability = float(min_probability)
        self._log_policy = np.full(
            (horizon, n_actions),
            -np.log(n_actions),
            dtype=np.float64,
        )

    @property
    def policy(self) -> FloatArray:
        return _softmax(self._log_policy)

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch = np.asarray(trajectories, dtype=np.int64)
        outcomes = np.asarray(rewards, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.horizon:
            raise ValueError(f"trajectories must have shape (batch, {self.horizon})")
        if outcomes.shape != (batch.shape[0],):
            raise ValueError("rewards must contain one value per trajectory")
        if batch.shape[0] < 2:
            raise ValueError("group-relative updates require at least two trajectories")
        if np.any(batch < 0) or np.any(batch >= self.n_actions):
            raise ValueError("trajectory contains an action outside the action space")
        if not np.all(np.isfinite(outcomes)):
            raise ValueError("rewards must be finite")

        old_policy = self.policy
        advantages = outcomes - outcomes.mean()
        reward_std = float(outcomes.std())
        if self.normalize_advantages and reward_std > 1e-12:
            advantages = advantages / reward_std

        action_scores = np.zeros_like(old_policy)
        group_size = batch.shape[0]
        for position in range(self.horizon):
            for action in range(self.n_actions):
                selected = batch[:, position] == action
                if not np.any(selected):
                    continue
                inverse_propensity = min(
                    1.0 / max(float(old_policy[position, action]), self.min_probability),
                    self.importance_clip,
                )
                action_scores[position, action] = (
                    float(advantages[selected].sum())
                    * inverse_propensity
                    / group_size
                )

        self._log_policy = np.log(np.maximum(old_policy, self.min_probability))
        self._log_policy += self.step_size * action_scores
        new_policy = _softmax(self._log_policy)
        new_policy = np.maximum(new_policy, self.min_probability)
        new_policy /= new_policy.sum(axis=1, keepdims=True)
        self._log_policy = np.log(new_policy)

        kl_drift = float(
            np.sum(
                new_policy
                * (
                    np.log(np.maximum(new_policy, self.min_probability))
                    - np.log(np.maximum(old_policy, self.min_probability))
                )
            )
        )

        return {
            "mean_reward": float(outcomes.mean()),
            "reward_std": reward_std,
            "zero_variance_group": float(reward_std <= 1e-12),
            "update_norm": float(np.linalg.norm(action_scores)),
            "kl_drift": kl_drift,
        }

    def entropy(self) -> float:
        probabilities = self.policy
        return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
