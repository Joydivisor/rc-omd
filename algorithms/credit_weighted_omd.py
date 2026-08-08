"""Step-sensitive OMD baselines for controlled credit experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .group_omd import FloatArray, GroupOMD, IntArray


def policy_entropy_by_position(policy: FloatArray) -> FloatArray:
    probabilities = np.asarray(policy, dtype=np.float64)
    return -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300)),
        axis=1,
    )


class EntropyWeightedOMD(GroupOMD):
    """Group OMD whose trajectory advantages are weighted by step entropy.

    Entropy weights are normalized to mean one. This keeps their average update
    scale comparable to Uniform Group OMD while changing where the update is
    allocated along the trajectory.
    """

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        advantages, reward_std = self._group_advantages(outcomes)
        entropies = policy_entropy_by_position(self.policy)
        mean_entropy = float(entropies.mean())
        if mean_entropy <= 1e-12:
            weights = np.ones(self.horizon, dtype=np.float64)
        else:
            weights = entropies / mean_entropy
        step_signals = advantages[:, None] * weights[None, :]
        stats = self._apply_step_signals(batch, step_signals)
        stats.update(
            {
                "mean_reward": float(outcomes.mean()),
                "reward_std": reward_std,
                "zero_variance_group": float(reward_std <= 1e-12),
                "min_step_weight": float(weights.min()),
                "max_step_weight": float(weights.max()),
            }
        )
        return stats


class OracleCreditOMD(GroupOMD):
    """Diagnostic OMD baseline using exact on-trajectory step credit.

    The credit signal is oracle-quality, but performance is not necessarily an
    upper bound because this baseline shares a fixed, untuned update scale with
    the other methods.
    """

    def update_with_credit(
        self,
        trajectories: IntArray,
        rewards: FloatArray,
        oracle_credits: NDArray[np.floating],
    ) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        credits = np.asarray(oracle_credits, dtype=np.float64)
        if credits.shape != batch.shape:
            raise ValueError("oracle_credits must have the same shape as trajectories")
        stats = self._apply_step_signals(batch, credits)
        reward_std = float(outcomes.std())
        stats.update(
            {
                "mean_reward": float(outcomes.mean()),
                "reward_std": reward_std,
                "zero_variance_group": float(reward_std <= 1e-12),
                "mean_absolute_credit": float(np.mean(np.abs(credits))),
            }
        )
        return stats
