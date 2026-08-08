"""Low-cost online reliability-calibrated OMD."""

from __future__ import annotations

from credit_estimators import RunningCreditEstimate, RunningMomentsCreditEstimator

from .group_omd import FloatArray, GroupOMD, IntArray


class OnlineReliabilityOMD(GroupOMD):
    """Control local OMD steps using running action-score consistency."""

    def __init__(
        self,
        *,
        reliability_decay: float = 0.9,
        confidence_multiplier: float = 1.0,
        warmup_effective_samples: float = 8.0,
        reliability_floor: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        self.reliability_floor = float(reliability_floor)
        self.credit_estimator = RunningMomentsCreditEstimator(
            decay=reliability_decay,
            confidence_multiplier=confidence_multiplier,
            warmup_effective_samples=warmup_effective_samples,
            importance_clip=self.importance_clip,
            min_probability=self.min_probability,
        )
        self.last_credit_estimate: RunningCreditEstimate | None = None
        self.last_reward_std: float | None = None

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        estimate = self.credit_estimator.estimate(batch, outcomes, self.policy)
        self.last_credit_estimate = estimate
        local_scales = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * estimate.reliability
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
                "effective_sample_size": estimate.effective_sample_size,
            }
        )
        return stats
