"""Threshold-reward sequence environments with exact counterfactual credit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class StructuredSequenceMDP:
    """Terminal reward when enough pivotal positions match their targets."""

    horizon: int
    n_actions: int
    critical_positions: tuple[int, ...]
    target_actions: tuple[int, ...]
    minimum_matches: int
    _target_by_position: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_actions <= 1:
            raise ValueError("n_actions must be at least two")
        if not self.critical_positions:
            raise ValueError("at least one critical position is required")
        if len(self.critical_positions) != len(self.target_actions):
            raise ValueError("critical_positions and target_actions must align")
        if len(set(self.critical_positions)) != len(self.critical_positions):
            raise ValueError("critical_positions must be unique")
        if not 1 <= self.minimum_matches <= len(self.critical_positions):
            raise ValueError("minimum_matches must be between one and the pivotal count")
        if any(position < 0 or position >= self.horizon for position in self.critical_positions):
            raise ValueError("critical position outside the horizon")
        if any(action < 0 or action >= self.n_actions for action in self.target_actions):
            raise ValueError("target action outside the action space")

        ordered = sorted(zip(self.critical_positions, self.target_actions, strict=True))
        object.__setattr__(self, "critical_positions", tuple(item[0] for item in ordered))
        object.__setattr__(self, "target_actions", tuple(item[1] for item in ordered))
        object.__setattr__(
            self,
            "_target_by_position",
            {position: action for position, action in ordered},
        )

    @classmethod
    def from_sequences(
        cls,
        *,
        horizon: int,
        n_actions: int,
        critical_positions: Sequence[int],
        target_actions: Sequence[int],
        minimum_matches: int,
    ) -> "StructuredSequenceMDP":
        return cls(
            horizon=horizon,
            n_actions=n_actions,
            critical_positions=tuple(int(value) for value in critical_positions),
            target_actions=tuple(int(value) for value in target_actions),
            minimum_matches=int(minimum_matches),
        )

    @property
    def distractor_positions(self) -> tuple[int, ...]:
        critical = set(self.critical_positions)
        return tuple(position for position in range(self.horizon) if position not in critical)

    def evaluate(self, actions: Sequence[int] | IntArray) -> float:
        trajectory = np.asarray(actions, dtype=np.int64)
        self._validate_trajectory(trajectory)
        matches = sum(
            int(trajectory[position] == target)
            for position, target in self._target_by_position.items()
        )
        return float(matches >= self.minimum_matches)

    def batch_rewards(self, trajectories: IntArray) -> FloatArray:
        batch = np.asarray(trajectories, dtype=np.int64)
        self._validate_batch(batch)
        matches = np.zeros(batch.shape[0], dtype=np.int64)
        for position, target in self._target_by_position.items():
            matches += batch[:, position] == target
        return (matches >= self.minimum_matches).astype(np.float64)

    def sample(
        self,
        policy: FloatArray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> IntArray:
        probabilities = self._validate_policy(policy)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        samples = np.empty((batch_size, self.horizon), dtype=np.int64)
        for position in range(self.horizon):
            samples[:, position] = rng.choice(
                self.n_actions,
                size=batch_size,
                p=probabilities[position],
            )
        return samples

    def expected_success_probability(self, policy: FloatArray) -> float:
        probabilities = self._validate_policy(policy)
        target_probabilities = [
            float(probabilities[position, target])
            for position, target in self._target_by_position.items()
        ]
        return self._completion_probability(0, target_probabilities)

    def oracle_step_credit(self, trajectory: IntArray, policy: FloatArray) -> FloatArray:
        actions = np.asarray(trajectory, dtype=np.int64)
        self._validate_trajectory(actions)
        probabilities = self._validate_policy(policy)
        credits = np.zeros(self.horizon, dtype=np.float64)
        prefix_matches = 0

        for position in range(self.horizon):
            target = self._target_by_position.get(position)
            if target is None:
                continue
            future_probabilities = [
                float(probabilities[future_position, future_target])
                for future_position, future_target in self._target_by_position.items()
                if future_position > position
            ]
            q_match = self._completion_probability(
                prefix_matches + 1,
                future_probabilities,
            )
            q_mismatch = self._completion_probability(
                prefix_matches,
                future_probabilities,
            )
            target_probability = float(probabilities[position, target])
            state_value = (
                target_probability * q_match
                + (1.0 - target_probability) * q_mismatch
            )
            q_taken = q_match if actions[position] == target else q_mismatch
            credits[position] = q_taken - state_value
            prefix_matches += int(actions[position] == target)
        return credits

    def oracle_batch_credit(self, trajectories: IntArray, policy: FloatArray) -> FloatArray:
        batch = np.asarray(trajectories, dtype=np.int64)
        self._validate_batch(batch)
        probabilities = self._validate_policy(policy)
        return np.stack(
            [self.oracle_step_credit(trajectory, probabilities) for trajectory in batch],
            axis=0,
        )

    def oracle_position_importance(self, policy: FloatArray) -> FloatArray:
        """Return unconditional expected absolute counterfactual credit."""

        probabilities = self._validate_policy(policy)
        importance = np.zeros(self.horizon, dtype=np.float64)
        prefix_distribution = np.asarray([1.0], dtype=np.float64)
        for position in range(self.horizon):
            target = self._target_by_position.get(position)
            if target is None:
                continue
            future_probabilities = [
                float(probabilities[future_position, future_target])
                for future_position, future_target in self._target_by_position.items()
                if future_position > position
            ]
            target_probability = float(probabilities[position, target])
            expected_absolute_credit = 0.0
            for matches, prefix_probability in enumerate(prefix_distribution):
                q_match = self._completion_probability(matches + 1, future_probabilities)
                q_mismatch = self._completion_probability(matches, future_probabilities)
                state_value = (
                    target_probability * q_match
                    + (1.0 - target_probability) * q_mismatch
                )
                expected_absolute_credit += float(prefix_probability) * (
                    target_probability * abs(q_match - state_value)
                    + (1.0 - target_probability) * abs(q_mismatch - state_value)
                )
            importance[position] = expected_absolute_credit
            prefix_distribution = self._extend_match_distribution(
                prefix_distribution,
                target_probability,
            )
        return importance

    def _completion_probability(
        self,
        prefix_matches: int,
        future_target_probabilities: Sequence[float],
    ) -> float:
        distribution = np.asarray([1.0], dtype=np.float64)
        for probability in future_target_probabilities:
            distribution = self._extend_match_distribution(distribution, probability)
        required_future = max(self.minimum_matches - prefix_matches, 0)
        if required_future <= 0:
            return 1.0
        if required_future >= len(distribution):
            return 0.0
        return float(distribution[required_future:].sum())

    @staticmethod
    def _extend_match_distribution(
        distribution: FloatArray,
        target_probability: float,
    ) -> FloatArray:
        extended = np.zeros(len(distribution) + 1, dtype=np.float64)
        extended[:-1] += distribution * (1.0 - target_probability)
        extended[1:] += distribution * target_probability
        return extended

    def _validate_trajectory(self, trajectory: IntArray) -> None:
        if trajectory.shape != (self.horizon,):
            raise ValueError(f"trajectory must have shape ({self.horizon},)")
        if np.any(trajectory < 0) or np.any(trajectory >= self.n_actions):
            raise ValueError("trajectory contains an action outside the action space")

    def _validate_batch(self, batch: IntArray) -> None:
        if batch.ndim != 2 or batch.shape[1] != self.horizon:
            raise ValueError(f"trajectories must have shape (batch, {self.horizon})")
        if np.any(batch < 0) or np.any(batch >= self.n_actions):
            raise ValueError("trajectory contains an action outside the action space")

    def _validate_policy(self, policy: FloatArray) -> FloatArray:
        probabilities = np.asarray(policy, dtype=np.float64)
        if probabilities.shape != (self.horizon, self.n_actions):
            raise ValueError(
                f"policy must have shape ({self.horizon}, {self.n_actions})"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("policy probabilities must be finite and non-negative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("each policy row must sum to one")
        return probabilities
