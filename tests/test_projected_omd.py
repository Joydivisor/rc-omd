"""Tests for shared-parameter projected mirror descent."""

from __future__ import annotations

import unittest

import numpy as np

from algorithms import ProjectedGroupOMD, ProjectedOnlineReliabilityOMD
from environments import StructuredSequenceMDP


class ProjectedOMDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = StructuredSequenceMDP.from_sequences(
            horizon=6,
            n_actions=2,
            critical_positions=[0, 2, 4],
            target_actions=[1, 0, 1],
            minimum_matches=2,
        )
        self.features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

    def _train(self, optimizer: ProjectedGroupOMD, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(120):
            trajectories = self.environment.sample(optimizer.policy, 64, rng)
            optimizer.update(
                trajectories,
                self.environment.batch_rewards(trajectories),
            )

    def test_shared_features_produce_shared_policy_rows(self) -> None:
        optimizer = ProjectedGroupOMD(
            horizon=6,
            n_actions=2,
            step_size=0.5,
            features=self.features,
        )
        self._train(optimizer)
        np.testing.assert_allclose(optimizer.policy[1], optimizer.policy[3])
        np.testing.assert_allclose(optimizer.policy[3], optimizer.policy[5])
        np.testing.assert_allclose(optimizer.policy.sum(axis=1), 1.0)

    def test_projected_group_omd_learns(self) -> None:
        optimizer = ProjectedGroupOMD(
            horizon=6,
            n_actions=2,
            step_size=0.5,
            features=self.features,
        )
        before = self.environment.expected_success_probability(optimizer.policy)
        self._train(optimizer)
        after = self.environment.expected_success_probability(optimizer.policy)
        self.assertGreater(after, before + 0.25)

    def test_projected_online_reliability_learns(self) -> None:
        optimizer = ProjectedOnlineReliabilityOMD(
            horizon=6,
            n_actions=2,
            step_size=0.8,
            features=self.features,
            reliability_decay=0.9,
            warmup_effective_samples=4.0,
        )
        before = self.environment.expected_success_probability(optimizer.policy)
        self._train(optimizer)
        after = self.environment.expected_success_probability(optimizer.policy)
        self.assertGreater(after, before + 0.20)
        self.assertIsNotNone(optimizer.last_credit_estimate)
        np.testing.assert_allclose(optimizer.policy.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
