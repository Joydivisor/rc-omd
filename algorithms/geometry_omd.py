"""Reliability-weighted cross-entropy projection OMD (geometry-v1 candidate).

Implements the objective locked in `docs/PARAMETER_SPACE_GEOMETRY_DESIGN.md`:

    L(theta) =   sum_k r_k          * D_KL( q_k      || pi_theta^k )
               + lambda * sum_k (1-r_k) * D_KL( pi_old^k || pi_theta^k )
               + (mu/2) * || theta - theta_old ||_F^2

Both divergences are forward KL, which keeps the projection subproblem convex
in `theta`. A positive combination of forward KLs against fixed references is a
single weighted cross-entropy, so the objective collapses exactly to

    sum_k w_k * CE( m_k , pi_theta^k )  +  const

    w_k = r_k + lambda * (1 - r_k)
    m_k = [ r_k * q_k + lambda (1-r_k) * pi_old^k ] / w_k

and it is that collapsed form which is implemented here. Weights are
mean-normalized so that `lambda` controls differential weighting without also
rescaling the gradient, the effective projection learning rate, the relative
strength of `mu`, or the meaning of the termination tolerance.

`mu` is the existing `projection_ridge` parameter; no separate knob is added.
"""

from __future__ import annotations

import numpy as np

from credit_estimators import RunningCreditEstimate, RunningMomentsCreditEstimator

from .group_omd import FloatArray, IntArray, _softmax
from .projected_omd import ProjectedGroupOMD


class ReliabilityWeightedProjectionOMD(ProjectedGroupOMD):
    """Projected OMD whose projection is weighted by per-position reliability.

    Differs from :class:`ProjectedOnlineReliabilityOMD` in two ways, both
    required by the design specification:

    * the OMD target ``q_k`` is built from the **raw** step size, with no
      reliability rescaling, so reliability enters the algorithm exactly once;
    * reliability enters the projection itself, through the mixture ``m_k`` and
      the mean-normalized weight ``w_k``, rather than only shaping the target.
    """

    def __init__(
        self,
        *,
        projection_lambda: float = 1.0,
        reliability_decay: float = 0.9,
        confidence_multiplier: float = 1.0,
        warmup_effective_samples: float = 8.0,
        reliability_floor: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(projection_lambda) or projection_lambda <= 0.0:
            raise ValueError("projection_lambda must be positive and finite")
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        # Values below 1 are permitted so the protocol can run its labelled
        # lambda < 1 negative check; the protocol, not this class, restricts
        # which values are selectable.
        self.projection_lambda = float(projection_lambda)
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

    def mixture_target(
        self,
        action_scores: FloatArray,
        reliability: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(q, m, w_tilde)`` for the current policy.

        Exposed separately from the projection so that the collapse identity
        and the degeneracy invariants can be tested without running gradient
        descent.
        """

        scores = np.asarray(action_scores, dtype=np.float64)
        if scores.shape != (self.horizon, self.n_actions):
            raise ValueError(
                f"action_scores must have shape ({self.horizon}, {self.n_actions})"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("action_scores must be finite")
        r = np.asarray(reliability, dtype=np.float64)
        if r.shape != (self.horizon,):
            raise ValueError(f"reliability must have shape ({self.horizon},)")
        if not np.all(np.isfinite(r)) or np.any(r < 0.0) or np.any(r > 1.0):
            raise ValueError("reliability must be finite and within [0, 1]")

        old_policy = self.policy
        log_old = np.log(np.maximum(old_policy, self.min_probability))
        target = _softmax(log_old + self.step_size * scores)

        lam = self.projection_lambda
        weights = r + lam * (1.0 - r)
        # weights >= min(1, lam) > 0, so the mean is strictly positive.
        normalized = weights / weights.mean()
        mixture = (
            r[:, None] * target + lam * (1.0 - r)[:, None] * old_policy
        ) / weights[:, None]
        mixture = np.maximum(mixture, self.min_probability)
        mixture /= mixture.sum(axis=1, keepdims=True)
        return target, mixture, normalized

    def _reliability_weighted_projection(
        self,
        action_scores: FloatArray,
        reliability: FloatArray,
    ) -> dict[str, float]:
        old_policy = self.policy
        log_old = np.log(np.maximum(old_policy, self.min_probability))
        _, mixture, normalized = self.mixture_target(action_scores, reliability)
        log_mixture = np.log(np.maximum(mixture, self.min_probability))
        old_weights = self.weights.copy()

        gradient_norm = float("inf")
        steps_used = 0
        for steps_used in range(1, self.projection_steps + 1):
            projected_policy = self.policy
            gradient = self.features.T @ (
                normalized[:, None] * (projected_policy - mixture)
            )
            gradient /= self.horizon
            gradient += self.projection_ridge * (self.weights - old_weights)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= self.projection_tolerance:
                break
            self.weights -= self.projection_learning_rate * gradient
            # Remove the softmax-invariant common offset for numerical stability.
            self.weights -= self.weights.mean(axis=1, keepdims=True)

        new_policy = self.policy
        log_new = np.log(np.maximum(new_policy, self.min_probability))

        # Design specification, Section 11: three distinct quantities.
        target_kl = np.sum(mixture * (log_mixture - log_old), axis=1)
        realized_kl = np.sum(new_policy * (log_new - log_old), axis=1)
        projection_residual = np.sum(mixture * (log_mixture - log_new), axis=1)

        scores = np.asarray(action_scores, dtype=np.float64)
        return {
            "update_norm": float(np.linalg.norm(normalized[:, None] * scores)),
            "kl_drift": float(realized_kl.sum()),
            "max_position_kl": float(realized_kl.max()),
            "parameter_update_norm": float(
                np.linalg.norm(self.weights - old_weights)
            ),
            "projection_kl": float(projection_residual.sum() / self.horizon),
            "projection_gradient_norm": gradient_norm,
            "projection_steps": float(steps_used),
            "target_kl_total": float(target_kl.sum()),
            "realized_kl_total": float(realized_kl.sum()),
            "projection_residual_total": float(projection_residual.sum()),
            "weight_spread": float(normalized.max() - normalized.min()),
        }

    def update(self, trajectories: IntArray, rewards: FloatArray) -> dict[str, float]:
        batch, outcomes = self._validate_group(trajectories, rewards)
        estimate = self.credit_estimator.estimate(batch, outcomes, self.policy)
        self.last_credit_estimate = estimate
        reliability = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * estimate.reliability
        stats = self._reliability_weighted_projection(
            estimate.action_scores, reliability
        )
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
