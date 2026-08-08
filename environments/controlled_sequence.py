"""A controlled delayed-reward sequence environment with exact step credit.

The agent selects one action at each position of a fixed-length sequence. Only
selected critical positions affect the terminal binary reward. This makes it
possible to construct high-entropy distractors and low-entropy pivotal actions
while retaining an exact counterfactual credit oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ControlledSequenceMDP:
    """Fixed-horizon terminal-reward environment.

    Reward is one exactly when every critical position matches its target
    action. Actions at non-critical positions are distractors and do not affect
    reward.
    """

    horizon: int
    n_actions: int
    critical_positions: tuple[int, ...]
    target_actions: tuple[int, ...]
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
    ) -> "ControlledSequenceMDP":
        return cls(
            horizon=horizon,
            n_actions=n_actions,
            critical_positions=tuple(int(value) for value in critical_positions),
            target_actions=tuple(int(value) for value in target_actions),
        )

    @property
    def distractor_positions(self) -> tuple[int, ...]:
        critical = set(self.critical_positions)
        return tuple(position for position in range(self.horizon) if position not in critical)

    def evaluate(self, actions: Sequence[int] | IntArray) -> float:
        trajectory = np.asarray(actions, dtype=np.int64)
        self._validate_trajectory(trajectory)
        return float(
            all(
                trajectory[position] == target
                for position, target in self._target_by_position.items()
            )
        )

    def batch_rewards(self, trajectories: IntArray) -> FloatArray:
        batch = np.asarray(trajectories, dtype=np.int64)
        if batch.ndim != 2 or batch.shape[1] != self.horizon:
            raise ValueError(f"trajectories must have shape (batch, {self.horizon})")
        if np.any(batch < 0) or np.any(batch >= self.n_actions):
            raise ValueError("trajectory contains an action outside the action space")

        success = np.ones(batch.shape[0], dtype=bool)
        for position, target in self._target_by_position.items():
            success &= batch[:, position] == target
        return success.astype(np.float64)

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
        success_probability = 1.0
        for position, target in self._target_by_position.items():
            success_probability *= float(probabilities[position, target])
        return success_probability

    def oracle_step_credit(self, trajectory: IntArray, policy: FloatArray) -> FloatArray:
        """Return exact on-trajectory counterfactual advantages.

        At step k, credit is Q(s_k, a_k) - V(s_k), conditioned on the sampled
        prefix. If an earlier critical action has already failed, future credit
        is zero because terminal success is no longer reachable.
        """

        actions = np.asarray(trajectory, dtype=np.int64)
        self._validate_trajectory(actions)
        probabilities = self._validate_policy(policy)

        credits = np.zeros(self.horizon, dtype=np.float64)
        viable_prefix = True

        for position in range(self.horizon):
            if not viable_prefix:
                break

            target = self._target_by_position.get(position)
            if target is None:
                continue

            future_success = 1.0
            for future_position, future_target in self._target_by_position.items():
                if future_position > position:
                    future_success *= float(probabilities[future_position, future_target])

            q_taken = future_success if actions[position] == target else 0.0
            state_value = float(probabilities[position, target]) * future_success
            credits[position] = q_taken - state_value

            if actions[position] != target:
                viable_prefix = False

        return credits

    def _validate_trajectory(self, trajectory: IntArray) -> None:
        if trajectory.shape != (self.horizon,):
            raise ValueError(f"trajectory must have shape ({self.horizon},)")
        if np.any(trajectory < 0) or np.any(trajectory >= self.n_actions):
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
