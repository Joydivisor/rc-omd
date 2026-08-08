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
        initial_policy: FloatArray | None = None,
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
        if initial_policy is None:
            probabilities = np.full(
                (horizon, n_actions),
                1.0 / n_actions,
                dtype=np.float64,
            )
        else:
            probabilities = np.asarray(initial_policy, dtype=np.float64)
            if probabilities.shape != (horizon, n_actions):
                raise ValueError(
                    f"initial_policy must have shape ({horizon}, {n_actions})"
                )
            if np.any(probabilities <= 0.0) or not np.all(np.isfinite(probabilities)):
                raise ValueError("initial_policy must contain finite positive values")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
                raise ValueError("each initial_policy row must sum to one")
        self._log_policy = np.log(probabilities)

    @property
    def policy(self) -> FloatArray:
        return _softmax(self._log_policy)

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        advantages, reward_std = self._group_advantages(outcomes)
        step_signals = np.repeat(advantages[:, None], self.horizon, axis=1)
        stats = self._apply_step_signals(batch, step_signals)
        stats.update(
            {
                "mean_reward": float(outcomes.mean()),
                "reward_std": reward_std,
                "zero_variance_group": float(reward_std <= 1e-12),
            }
        )
        return stats

    def _validate_group(
        self,
        trajectories: IntArray,
        rewards: FloatArray,
    ) -> tuple[IntArray, FloatArray]:
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
        return batch, outcomes

    def _group_advantages(self, outcomes: FloatArray) -> tuple[FloatArray, float]:
        advantages = outcomes - outcomes.mean()
        reward_std = float(outcomes.std())
        if self.normalize_advantages and reward_std > 1e-12:
            advantages = advantages / reward_std
        return advantages, reward_std

    def _apply_step_signals(
        self,
        trajectories: IntArray,
        step_signals: FloatArray,
    ) -> dict[str, float]:
        batch = np.asarray(trajectories, dtype=np.int64)
        signals = np.asarray(step_signals, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.horizon:
            raise ValueError(f"trajectories must have shape (batch, {self.horizon})")
        if signals.shape != batch.shape:
            raise ValueError("step_signals must have the same shape as trajectories")
        if not np.all(np.isfinite(signals)):
            raise ValueError("step_signals must be finite")

        action_scores = self._estimate_action_scores(batch, signals)
        return self._apply_action_scores(action_scores)

    def _estimate_action_scores(
        self,
        trajectories: IntArray,
        step_signals: FloatArray,
    ) -> FloatArray:
        """Estimate linearized per-action gains with inverse propensities."""

        batch = np.asarray(trajectories, dtype=np.int64)
        signals = np.asarray(step_signals, dtype=np.float64)
        old_policy = self.policy
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
                    float(signals[selected, position].sum())
                    * inverse_propensity
                    / group_size
                )
        return action_scores

    def _apply_action_scores(
        self,
        action_scores: FloatArray,
        local_step_scales: FloatArray | None = None,
    ) -> dict[str, float]:
        """Apply a KL-mirror update with optional per-position step scales."""

        scores = np.asarray(action_scores, dtype=np.float64)
        if scores.shape != (self.horizon, self.n_actions):
            raise ValueError(
                f"action_scores must have shape ({self.horizon}, {self.n_actions})"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("action_scores must be finite")

        if local_step_scales is None:
            scales = np.ones(self.horizon, dtype=np.float64)
        else:
            scales = np.asarray(local_step_scales, dtype=np.float64)
            if scales.shape != (self.horizon,):
                raise ValueError(f"local_step_scales must have shape ({self.horizon},)")
            if not np.all(np.isfinite(scales)) or np.any(scales < 0.0):
                raise ValueError("local_step_scales must be finite and non-negative")

        old_policy = self.policy
        self._log_policy = np.log(np.maximum(old_policy, self.min_probability))
        self._log_policy += self.step_size * scales[:, None] * scores
        new_policy = _softmax(self._log_policy)
        new_policy = np.maximum(new_policy, self.min_probability)
        new_policy /= new_policy.sum(axis=1, keepdims=True)
        self._log_policy = np.log(new_policy)

        per_position_kl = np.sum(
            new_policy
            * (
                np.log(np.maximum(new_policy, self.min_probability))
                - np.log(np.maximum(old_policy, self.min_probability))
            ),
            axis=1,
        )
        kl_drift = float(np.sum(per_position_kl))

        return {
            "update_norm": float(np.linalg.norm(scales[:, None] * scores)),
            "kl_drift": kl_drift,
            "max_position_kl": float(np.max(per_position_kl)),
        }

    def entropy(self) -> float:
        probabilities = self.policy
        return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
