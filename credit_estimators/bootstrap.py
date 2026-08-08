"""Bootstrap reliability estimates for group-relative trajectory credit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class BootstrapCreditEstimate:
    """Linearized credit scores and their position-level reliability."""

    action_scores: FloatArray
    action_score_std: FloatArray
    reliability: FloatArray
    signal_norm: FloatArray
    uncertainty_norm: FloatArray


class BootstrapCreditEstimator:
    """Estimate whether empirical credit directions survive resampling.

    Reliability is a confidence-shrinkage factor

        q_k = max(0, 1 - z * u_k / (s_k + eps)),

    where s_k is the norm of the full-batch action score and u_k is the norm
    of its bootstrap standard error. A position receives a non-zero local step
    only when its estimated signal exceeds the configured uncertainty margin.
    """

    def __init__(
        self,
        *,
        n_bootstrap: int = 64,
        confidence_multiplier: float = 1.96,
        importance_clip: float = 20.0,
        min_probability: float = 1e-8,
        seed: int = 0,
    ) -> None:
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least two")
        if confidence_multiplier < 0.0:
            raise ValueError("confidence_multiplier must be non-negative")
        if importance_clip <= 0.0:
            raise ValueError("importance_clip must be positive")
        if min_probability <= 0.0:
            raise ValueError("min_probability must be positive")
        self.n_bootstrap = int(n_bootstrap)
        self.confidence_multiplier = float(confidence_multiplier)
        self.importance_clip = float(importance_clip)
        self.min_probability = float(min_probability)
        self.rng = np.random.default_rng(seed)

    def estimate(
        self,
        trajectories: IntArray,
        rewards: FloatArray,
        policy: FloatArray,
    ) -> BootstrapCreditEstimate:
        batch = np.asarray(trajectories, dtype=np.int64)
        outcomes = np.asarray(rewards, dtype=np.float64)
        probabilities = np.asarray(policy, dtype=np.float64)
        if batch.ndim != 2:
            raise ValueError("trajectories must be a two-dimensional array")
        group_size, horizon = batch.shape
        if group_size < 2:
            raise ValueError("bootstrap credit requires at least two trajectories")
        if outcomes.shape != (group_size,):
            raise ValueError("rewards must contain one value per trajectory")
        if probabilities.ndim != 2 or probabilities.shape[0] != horizon:
            raise ValueError("policy horizon must match trajectories")
        n_actions = probabilities.shape[1]
        if np.any(batch < 0) or np.any(batch >= n_actions):
            raise ValueError("trajectory contains an action outside the policy")
        if not np.all(np.isfinite(outcomes)):
            raise ValueError("rewards must be finite")

        advantages = outcomes - outcomes.mean()
        full_scores = self._scores(batch, advantages, probabilities)
        bootstrap_scores = np.empty(
            (self.n_bootstrap, horizon, n_actions),
            dtype=np.float64,
        )
        for bootstrap_index in range(self.n_bootstrap):
            indices = self.rng.integers(0, group_size, size=group_size)
            sampled_rewards = outcomes[indices]
            sampled_advantages = sampled_rewards - sampled_rewards.mean()
            bootstrap_scores[bootstrap_index] = self._scores(
                batch[indices],
                sampled_advantages,
                probabilities,
            )

        score_std = bootstrap_scores.std(axis=0, ddof=1)
        centered_scores = full_scores - full_scores.mean(axis=1, keepdims=True)
        signal_norm = np.linalg.norm(centered_scores, axis=1)
        uncertainty_norm = np.linalg.norm(score_std, axis=1)
        reliability = np.where(
            signal_norm > 1e-12,
            np.maximum(
                0.0,
                1.0
                - self.confidence_multiplier
                * uncertainty_norm
                / np.maximum(signal_norm, 1e-12),
            ),
            0.0,
        )
        reliability = np.minimum(reliability, 1.0)
        return BootstrapCreditEstimate(
            action_scores=full_scores,
            action_score_std=score_std,
            reliability=reliability,
            signal_norm=signal_norm,
            uncertainty_norm=uncertainty_norm,
        )

    def _scores(
        self,
        trajectories: IntArray,
        advantages: FloatArray,
        policy: FloatArray,
    ) -> FloatArray:
        group_size, horizon = trajectories.shape
        n_actions = policy.shape[1]
        scores = np.zeros((horizon, n_actions), dtype=np.float64)
        for position in range(horizon):
            for action in range(n_actions):
                selected = trajectories[:, position] == action
                if not np.any(selected):
                    continue
                inverse_propensity = min(
                    1.0 / max(float(policy[position, action]), self.min_probability),
                    self.importance_clip,
                )
                scores[position, action] = (
                    float(advantages[selected].sum())
                    * inverse_propensity
                    / group_size
                )
        return scores
