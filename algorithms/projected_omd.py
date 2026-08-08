"""Approximate mirror descent under shared linear policy parameters."""

from __future__ import annotations

import numpy as np

from credit_estimators import RunningCreditEstimate, RunningMomentsCreditEstimator

from .group_omd import FloatArray, GroupOMD, IntArray, _softmax


class ProjectedGroupOMD(GroupOMD):
    """Group OMD followed by a KL projection into a shared linear policy.

    The policy logits are ``features @ weights``.  A tabular OMD target is first
    constructed at every position and then projected into this shared parameter
    family by minimizing target-to-parametric KL with gradient descent.
    """

    def __init__(
        self,
        *,
        features: FloatArray,
        projection_steps: int = 60,
        projection_learning_rate: float = 0.5,
        projection_ridge: float = 0.0,
        projection_tolerance: float = 1e-9,
        **kwargs: object,
    ) -> None:
        requested_initial_policy = kwargs.get("initial_policy")
        if requested_initial_policy is not None:
            initial = np.asarray(requested_initial_policy, dtype=np.float64)
            uniform = np.full_like(initial, 1.0 / int(kwargs["n_actions"]))
            if not np.allclose(initial, uniform, atol=1e-12):
                raise ValueError(
                    "projected OMD currently supports only a uniform initial policy"
                )
        super().__init__(**kwargs)
        feature_matrix = np.asarray(features, dtype=np.float64)
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] != self.horizon:
            raise ValueError("features must have shape (horizon, feature_dimension)")
        if feature_matrix.shape[1] <= 0:
            raise ValueError("feature_dimension must be positive")
        if not np.all(np.isfinite(feature_matrix)):
            raise ValueError("features must be finite")
        if projection_steps <= 0:
            raise ValueError("projection_steps must be positive")
        if projection_learning_rate <= 0.0:
            raise ValueError("projection_learning_rate must be positive")
        if projection_ridge < 0.0:
            raise ValueError("projection_ridge must be non-negative")
        if projection_tolerance < 0.0:
            raise ValueError("projection_tolerance must be non-negative")

        self.features = feature_matrix
        self.projection_steps = int(projection_steps)
        self.projection_learning_rate = float(projection_learning_rate)
        self.projection_ridge = float(projection_ridge)
        self.projection_tolerance = float(projection_tolerance)
        self.weights = np.zeros(
            (feature_matrix.shape[1], self.n_actions),
            dtype=np.float64,
        )

    @property
    def policy(self) -> FloatArray:
        return _softmax(self.features @ self.weights)

    def _apply_action_scores(
        self,
        action_scores: FloatArray,
        local_step_scales: FloatArray | None = None,
    ) -> dict[str, float]:
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
        target_logits = np.log(np.maximum(old_policy, self.min_probability))
        target_logits += self.step_size * scales[:, None] * scores
        target_policy = _softmax(target_logits)
        old_weights = self.weights.copy()

        gradient_norm = float("inf")
        steps_used = 0
        for steps_used in range(1, self.projection_steps + 1):
            projected_policy = self.policy
            gradient = self.features.T @ (projected_policy - target_policy)
            gradient /= self.horizon
            gradient += self.projection_ridge * (self.weights - old_weights)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= self.projection_tolerance:
                break
            self.weights -= self.projection_learning_rate * gradient
            # Remove the softmax-invariant common offset for numerical stability.
            self.weights -= self.weights.mean(axis=1, keepdims=True)

        new_policy = self.policy
        per_position_kl = np.sum(
            new_policy
            * (
                np.log(np.maximum(new_policy, self.min_probability))
                - np.log(np.maximum(old_policy, self.min_probability))
            ),
            axis=1,
        )
        target_projection_kl = np.sum(
            target_policy
            * (
                np.log(np.maximum(target_policy, self.min_probability))
                - np.log(np.maximum(new_policy, self.min_probability))
            )
        ) / self.horizon

        return {
            "update_norm": float(np.linalg.norm(scales[:, None] * scores)),
            "kl_drift": float(per_position_kl.sum()),
            "max_position_kl": float(per_position_kl.max()),
            "parameter_update_norm": float(np.linalg.norm(self.weights - old_weights)),
            "projection_kl": float(target_projection_kl),
            "projection_gradient_norm": gradient_norm,
            "projection_steps": float(steps_used),
        }


class ProjectedOnlineReliabilityOMD(ProjectedGroupOMD):
    """Projected OMD whose tabular target uses online reliability scales."""

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
