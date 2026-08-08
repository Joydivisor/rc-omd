from __future__ import annotations

import unittest

import numpy as np

from algorithms import (
    EntropyWeightedOMD,
    GroupOMD,
    OracleCreditOMD,
    ReliabilityCalibratedOMD,
)
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

    def test_entropy_weighting_changes_update_allocation(self) -> None:
        initial_policy = np.asarray(
            [
                [0.5, 0.5],
                [0.99, 0.01],
            ]
        )
        algorithm = EntropyWeightedOMD(
            horizon=2,
            n_actions=2,
            step_size=0.5,
            initial_policy=initial_policy,
        )
        trajectories = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        rewards = np.asarray([1.0, 1.0, 0.0, 0.0])
        stats = algorithm.update(trajectories, rewards)
        self.assertGreater(stats["max_step_weight"], stats["min_step_weight"])

    def test_oracle_credit_update_does_not_move_distractor_policy(self) -> None:
        environment = ControlledSequenceMDP(
            horizon=3,
            n_actions=2,
            critical_positions=(0, 2),
            target_actions=(1, 0),
        )
        algorithm = OracleCreditOMD(horizon=3, n_actions=2, step_size=0.5)
        rng = np.random.default_rng(4)
        old_distractor = algorithm.policy[1].copy()
        trajectories = environment.sample(algorithm.policy, 64, rng)
        rewards = environment.batch_rewards(trajectories)
        credits = environment.oracle_batch_credit(trajectories, algorithm.policy)
        algorithm.update_with_credit(trajectories, rewards, credits)
        np.testing.assert_allclose(algorithm.policy[1], old_distractor)

    def test_reliability_calibrated_omd_learns_and_preserves_simplex(self) -> None:
        environment = ControlledSequenceMDP(
            horizon=4,
            n_actions=2,
            critical_positions=(1, 3),
            target_actions=(1, 0),
        )
        algorithm = ReliabilityCalibratedOMD(
            horizon=4,
            n_actions=2,
            step_size=0.5,
            bootstrap_samples=24,
            confidence_multiplier=1.0,
            estimator_seed=5,
        )
        rng = np.random.default_rng(19)
        initial_success = environment.expected_success_probability(algorithm.policy)

        for _ in range(150):
            trajectories = environment.sample(algorithm.policy, 64, rng)
            rewards = environment.batch_rewards(trajectories)
            algorithm.update(trajectories, rewards)

        np.testing.assert_allclose(algorithm.policy.sum(axis=1), 1.0)
        self.assertTrue(np.all(algorithm.policy > 0.0))
        self.assertGreater(
            environment.expected_success_probability(algorithm.policy),
            initial_success + 0.5,
        )


if __name__ == "__main__":
    unittest.main()
