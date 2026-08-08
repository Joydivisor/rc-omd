from __future__ import annotations

import unittest

import numpy as np

from algorithms import GroupOMD
from environments import ControlledSequenceMDP


class GroupOMDTest(unittest.TestCase):
    def test_policy_remains_a_probability_distribution(self) -> None:
        algorithm = GroupOMD(horizon=2, n_actions=2, step_size=0.5)
        trajectories = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        rewards = np.asarray([1.0, 1.0, 0.0, 0.0])
        stats = algorithm.update(trajectories, rewards)

        self.assertTrue(np.all(algorithm.policy > 0.0))
        np.testing.assert_allclose(algorithm.policy.sum(axis=1), np.ones(2))
        self.assertGreaterEqual(stats["kl_drift"], 0.0)

    def test_learning_improves_exact_success_probability(self) -> None:
        environment = ControlledSequenceMDP(
            horizon=4,
            n_actions=2,
            critical_positions=(1, 3),
            target_actions=(1, 0),
        )
        algorithm = GroupOMD(horizon=4, n_actions=2, step_size=0.5)
        rng = np.random.default_rng(11)
        initial_success = environment.expected_success_probability(algorithm.policy)

        for _ in range(250):
            trajectories = environment.sample(algorithm.policy, 32, rng)
            rewards = environment.batch_rewards(trajectories)
            algorithm.update(trajectories, rewards)

        final_success = environment.expected_success_probability(algorithm.policy)
        self.assertGreater(final_success, initial_success + 0.5)
        self.assertGreater(final_success, 0.85)


if __name__ == "__main__":
    unittest.main()
