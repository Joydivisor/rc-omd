from __future__ import annotations

import unittest

import numpy as np

from credit_estimators import BootstrapCreditEstimator
from environments import ControlledSequenceMDP


class BootstrapCreditEstimatorTest(unittest.TestCase):
    def test_reliability_separates_pivotal_and_distractor_positions(self) -> None:
        environment = ControlledSequenceMDP(
            horizon=6,
            n_actions=2,
            critical_positions=(1, 4),
            target_actions=(1, 0),
        )
        policy = np.full((6, 2), 0.5)
        rng = np.random.default_rng(13)
        trajectories = environment.sample(policy, 4096, rng)
        rewards = environment.batch_rewards(trajectories)
        estimator = BootstrapCreditEstimator(n_bootstrap=48, seed=7)

        estimate = estimator.estimate(trajectories, rewards, policy)

        self.assertGreater(float(estimate.reliability[1]), 0.8)
        self.assertGreater(float(estimate.reliability[4]), 0.8)
        self.assertLess(float(np.max(estimate.reliability[[0, 2, 3, 5]])), 0.2)

    def test_zero_variance_rewards_produce_zero_reliability(self) -> None:
        policy = np.full((3, 2), 0.5)
        trajectories = np.asarray(
            [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
            dtype=np.int64,
        )
        rewards = np.zeros(4)
        estimator = BootstrapCreditEstimator(n_bootstrap=16, seed=2)

        estimate = estimator.estimate(trajectories, rewards, policy)

        np.testing.assert_allclose(estimate.action_scores, 0.0)
        np.testing.assert_allclose(estimate.reliability, 0.0)


if __name__ == "__main__":
    unittest.main()
