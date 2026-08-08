"""Reliability-calibrated local-step Online Mirror Descent."""

from __future__ import annotations

import numpy as np

from credit_estimators import BootstrapCreditEstimate, BootstrapCreditEstimator

from .group_omd import FloatArray, GroupOMD, IntArray


class ReliabilityCalibratedOMD(GroupOMD):
    """Use bootstrap credit reliability to control local OMD step sizes."""

    def __init__(
        self,
        *,
        bootstrap_samples: int = 64,
        confidence_multiplier: float = 1.96,
        reliability_floor: float = 0.0,
        estimator_seed: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        self.reliability_floor = float(reliability_floor)
        self.credit_estimator = BootstrapCreditEstimator(
            n_bootstrap=bootstrap_samples,
            confidence_multiplier=confidence_multiplier,
            importance_clip=self.importance_clip,
            min_probability=self.min_probability,
            seed=estimator_seed,
        )
        self.last_credit_estimate: BootstrapCreditEstimate | None = None
        self.last_reward_std: float | None = None

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        estimate = self.credit_estimator.estimate(batch, outcomes, self.policy)
        self.last_credit_estimate = estimate
        local_scales = self._step_scales(estimate)
        stats = self._apply_action_scores(estimate.action_scores, local_scales)
        reward_std = float(outcomes.std())
        self.last_reward_std = reward_std
        stats.update(
            {
                "mean_reward": float(outcomes.mean()),
                "reward_std": reward_std,
                "zero_variance_group": float(reward_std <= 1e-12),
                "mean_reliability": float(estimate.reliability.mean()),
                "min_reliability": float(estimate.reliability.min()),
                "max_reliability": float(estimate.reliability.max()),
                "mean_signal_norm": float(estimate.signal_norm.mean()),
                "mean_uncertainty_norm": float(estimate.uncertainty_norm.mean()),
            }
        )
        return stats

    def _step_scales(self, estimate: BootstrapCreditEstimate) -> FloatArray:
        return self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * estimate.reliability


class GlobalReliabilityOMD(ReliabilityCalibratedOMD):
    """Ablation using one batch-level reliability scale at every position."""

    def _step_scales(self, estimate: BootstrapCreditEstimate) -> FloatArray:
        global_reliability = float(estimate.reliability.max())
        scale = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * global_reliability
        return np.full(self.horizon, scale, dtype=np.float64)
