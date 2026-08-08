from __future__ import annotations

import unittest

import numpy as np

from environments import StructuredSequenceMDP


class StructuredSequenceMDPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = StructuredSequenceMDP(
            horizon=4,
            n_actions=2,
            critical_positions=(0, 1, 3),
            target_actions=(1, 1, 1),
            minimum_matches=2,
        )

    def test_threshold_reward(self) -> None:
        self.assertEqual(self.environment.evaluate([1, 1, 0, 0]), 1.0)
        self.assertEqual(self.environment.evaluate([1, 0, 0, 1]), 1.0)
        self.assertEqual(self.environment.evaluate([1, 0, 1, 0]), 0.0)

    def test_expected_probability_matches_binomial_case(self) -> None:
        policy = np.full((4, 2), 0.5)
        self.assertAlmostEqual(
            self.environment.expected_success_probability(policy),
            0.5,
        )

    def test_oracle_credit_matches_hand_calculation(self) -> None:
        environment = StructuredSequenceMDP(
            horizon=3,
            n_actions=2,
            critical_positions=(0, 1, 2),
            target_actions=(1, 1, 1),
            minimum_matches=2,
        )
        policy = np.full((3, 2), 0.5)
        credit = environment.oracle_step_credit(np.asarray([1, 0, 1]), policy)
        np.testing.assert_allclose(credit, np.asarray([0.25, -0.25, 0.5]))

    def test_distractor_has_zero_oracle_importance(self) -> None:
        policy = np.full((4, 2), 0.5)
        importance = self.environment.oracle_position_importance(policy)
        self.assertEqual(importance[2], 0.0)
        self.assertTrue(np.all(importance[[0, 1, 3]] > 0.0))


if __name__ == "__main__":
    unittest.main()
