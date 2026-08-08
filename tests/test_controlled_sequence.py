from __future__ import annotations

import unittest

import numpy as np

from environments import ControlledSequenceMDP


class ControlledSequenceMDPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = ControlledSequenceMDP(
            horizon=3,
            n_actions=2,
            critical_positions=(0, 2),
            target_actions=(1, 0),
        )

    def test_distractor_actions_do_not_change_reward(self) -> None:
        self.assertEqual(self.environment.evaluate([1, 0, 0]), 1.0)
        self.assertEqual(self.environment.evaluate([1, 1, 0]), 1.0)
        self.assertEqual(self.environment.evaluate([0, 0, 0]), 0.0)

    def test_expected_success_probability_is_exact(self) -> None:
        policy = np.asarray([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        self.assertAlmostEqual(
            self.environment.expected_success_probability(policy),
            0.5 * 0.7,
        )

    def test_oracle_credit_matches_counterfactual_advantage(self) -> None:
        policy = np.asarray([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        successful = self.environment.oracle_step_credit(
            np.asarray([1, 1, 0]),
            policy,
        )
        np.testing.assert_allclose(successful, np.asarray([0.35, 0.0, 0.3]))

        failed_early = self.environment.oracle_step_credit(
            np.asarray([0, 0, 0]),
            policy,
        )
        np.testing.assert_allclose(failed_early, np.asarray([-0.35, 0.0, 0.0]))

    def test_batch_rewards(self) -> None:
        trajectories = np.asarray(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(
            self.environment.batch_rewards(trajectories),
            np.asarray([1.0, 1.0, 0.0, 0.0]),
        )

    def test_oracle_position_importance_is_zero_for_distractors(self) -> None:
        policy = np.asarray([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        importance = self.environment.oracle_position_importance(policy)
        self.assertEqual(importance[1], 0.0)
        self.assertGreater(importance[0], 0.0)
        self.assertGreater(importance[2], 0.0)

    def test_batch_credit_matches_individual_credit(self) -> None:
        policy = np.asarray([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        trajectories = np.asarray([[1, 1, 0], [0, 0, 0]])
        expected = np.stack(
            [
                self.environment.oracle_step_credit(trajectory, policy)
                for trajectory in trajectories
            ]
        )
        np.testing.assert_allclose(
            self.environment.oracle_batch_credit(trajectories, policy),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
