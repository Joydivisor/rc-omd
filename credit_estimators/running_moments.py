"""Low-cost online reliability from exponentially weighted score moments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scoring import inverse_propensity_action_scores


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class RunningCreditEstimate:
    action_scores: FloatArray
    action_score_std: FloatArray
    reliability: FloatArray
    signal_norm: FloatArray
    uncertainty_norm: FloatArray
    effective_sample_size: float


class RunningMomentsCreditEstimator:
    """Track persistent credit directions with exponentially weighted moments.

    A true credit direction should recur across groups, while finite-sample
    distractor correlations should fluctuate and average out. The estimator
    uses O(H A) state and one score computation per update.
    """

    def __init__(
        self,
        *,
        decay: float = 0.9,
        confidence_multiplier: float = 1.0,
        warmup_effective_samples: float = 8.0,
        importance_clip: float = 20.0,
        min_probability: float = 1e-8,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if confidence_multiplier < 0.0:
            raise ValueError("confidence_multiplier must be non-negative")
        if warmup_effective_samples <= 0.0:
            raise ValueError("warmup_effective_samples must be positive")
        self.decay = float(decay)
        self.confidence_multiplier = float(confidence_multiplier)
        self.warmup_effective_samples = float(warmup_effective_samples)
        self.importance_clip = float(importance_clip)
        self.min_probability = float(min_probability)
        self._mean_numerator: FloatArray | None = None
        self._second_numerator: FloatArray | None = None
        self._weight_sum = 0.0
        self._squared_weight_sum = 0.0

    def estimate(
        self,
        trajectories: IntArray,
        rewards: FloatArray,
        policy: FloatArray,
    ) -> RunningCreditEstimate:
        outcomes = np.asarray(rewards, dtype=np.float64)
        advantages = outcomes - outcomes.mean()
        action_scores = inverse_propensity_action_scores(
            trajectories,
            advantages,
            policy,
            importance_clip=self.importance_clip,
            min_probability=self.min_probability,
        )
        centered_scores = action_scores - action_scores.mean(axis=1, keepdims=True)

        new_weight = 1.0 - self.decay
        if self._mean_numerator is None:
            self._mean_numerator = np.zeros_like(centered_scores)
            self._second_numerator = np.zeros_like(centered_scores)
        if self._second_numerator is None:
            raise RuntimeError("running second moment was not initialized")
        self._mean_numerator = (
            self.decay * self._mean_numerator + new_weight * centered_scores
        )
        self._second_numerator = (
            self.decay * self._second_numerator
            + new_weight * np.square(centered_scores)
        )
        self._weight_sum = self.decay * self._weight_sum + new_weight
        self._squared_weight_sum = (
            self.decay**2 * self._squared_weight_sum + new_weight**2
        )

        mean = self._mean_numerator / max(self._weight_sum, 1e-12)
        second = self._second_numerator / max(self._weight_sum, 1e-12)
        variance = np.maximum(second - np.square(mean), 0.0)
        effective_sample_size = self._weight_sum**2 / max(
            self._squared_weight_sum,
            1e-12,
        )
        standard_error = np.sqrt(variance / max(effective_sample_size, 1.0))
        signal_norm = np.linalg.norm(mean, axis=1)
        uncertainty_norm = np.linalg.norm(standard_error, axis=1)
        confidence = np.where(
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
        maturity = min(
            1.0,
            effective_sample_size / self.warmup_effective_samples,
        )
        reliability = np.minimum(confidence * maturity, 1.0)
        return RunningCreditEstimate(
            action_scores=action_scores,
            action_score_std=np.sqrt(variance),
            reliability=reliability,
            signal_norm=signal_norm,
            uncertainty_norm=uncertainty_norm,
            effective_sample_size=float(effective_sample_size),
        )
