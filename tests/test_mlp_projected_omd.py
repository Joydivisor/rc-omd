"""Numerical invariants for the Stage B non-linear parameterization.

The Jacobian check is the load-bearing one: the continuous `alpha` estimator is
built entirely from per-position parameter Jacobians, so an error there would
silently corrupt the predictor rather than crash.
"""

from __future__ import annotations

import unittest

import numpy as np

from algorithms import (
    MLPHead,
    MLPProjectedGroupOMD,
    MLPReliabilityWeightedProjectionOMD,
)
from environments import StructuredSequenceMDP


def make_features(horizon: int, groups: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((horizon, len(groups)))
    for column, members in enumerate(groups):
        for position in members:
            matrix[position, column] = 1.0
    return matrix


GROUPS = [[0, 1], [2, 3, 4], [5, 6], [7]]
HORIZON, N_ACTIONS = 8, 3
FEATURES = make_features(HORIZON, GROUPS)

BASE = {
    "horizon": HORIZON,
    "n_actions": N_ACTIONS,
    "step_size": 0.5,
    "normalize_advantages": False,
    "importance_clip": 20.0,
    "initial_policy": np.full((HORIZON, N_ACTIONS), 1.0 / N_ACTIONS),
    "features": FEATURES,
    "projection_steps": 40,
    "projection_learning_rate": 0.5,
    "projection_ridge": 0.0,
    "projection_tolerance": 1e-12,
}
RELIABILITY = {
    "reliability_decay": 0.9,
    "confidence_multiplier": 1.0,
    "warmup_effective_samples": 8.0,
    "reliability_floor": 0.1,
}


class MLPHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.head = MLPHead(feature_dim=4, n_actions=3, hidden=6, seed=3)
        rng = np.random.default_rng(0)
        # Move off the zero-output initialization so the Jacobian is non-trivial.
        self.head.W2 = rng.normal(size=self.head.W2.shape)
        self.head.b2 = rng.normal(size=self.head.b2.shape)
        self.features = rng.normal(size=(5, 4))

    def test_zero_output_layer_gives_uniform_policy(self) -> None:
        fresh = MLPHead(feature_dim=4, n_actions=3, hidden=6, seed=11)
        logits = fresh.logits(self.features)
        np.testing.assert_allclose(logits, 0.0, atol=1e-15)

    def test_position_jacobian_matches_finite_differences(self) -> None:
        """The estimator's entire input; an error here corrupts alpha silently."""

        analytic = self.head.position_jacobians(self.features)
        n_params = self.head.n_parameters
        shapes = [
            ("W1", self.head.W1.shape, self.head.W1.size),
            ("b1", self.head.b1.shape, self.head.b1.size),
            ("W2", self.head.W2.shape, self.head.W2.size),
            ("b2", self.head.b2.shape, self.head.b2.size),
        ]
        epsilon = 1e-6
        worst = 0.0
        for position in (0, 3):
            for action in range(self.head.n_actions):
                offset = 0
                for name, shape, size in shapes:
                    array = getattr(self.head, name)
                    flat = array.ravel()
                    for index in range(size):
                        original = flat[index]
                        flat[index] = original + epsilon
                        up = self.head.logits(self.features)[position, action]
                        flat[index] = original - epsilon
                        down = self.head.logits(self.features)[position, action]
                        flat[index] = original
                        numeric = (up - down) / (2 * epsilon)
                        got = analytic[
                            position, action * n_params + offset + index
                        ]
                        worst = max(worst, abs(numeric - got))
                    offset += size
        self.assertLess(worst, 1e-7, f"max Jacobian error {worst:.3e}")

    def test_backprop_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(5)
        target = rng.normal(size=(5, 3))

        def loss() -> float:
            return float(np.sum(self.head.logits(self.features) * target))

        gW1, gb1, gW2, gb2 = self.head.parameter_gradient(self.features, target)
        epsilon = 1e-6
        worst = 0.0
        for array, gradient in (
            (self.head.W1, gW1), (self.head.b1, gb1),
            (self.head.W2, gW2), (self.head.b2, gb2),
        ):
            flat, gflat = array.ravel(), np.asarray(gradient).ravel()
            for index in range(flat.size):
                original = flat[index]
                flat[index] = original + epsilon
                up = loss()
                flat[index] = original - epsilon
                down = loss()
                flat[index] = original
                worst = max(worst, abs((up - down) / (2 * epsilon) - gflat[index]))
        self.assertLess(worst, 1e-7, f"max backprop error {worst:.3e}")

    def test_linear_jacobian_gram_is_exactly_the_tie_group_indicator(self) -> None:
        """The calibration anchor: the estimator must be exact where alpha is defined.

        For a softmax-linear head the per-position Jacobian is determined by
        ``f_k``, so cosine similarity reduces to the one-hot group indicator and
        the continuous alpha collapses onto the discrete one.
        """

        normalized = FEATURES / np.linalg.norm(FEATURES, axis=1, keepdims=True)
        similarity = np.clip(normalized @ normalized.T, 0.0, None)
        expected = np.zeros((HORIZON, HORIZON))
        for members in GROUPS:
            for j in members:
                for k in members:
                    expected[j, k] = 1.0
        np.testing.assert_allclose(similarity, expected, atol=1e-12)

    def test_mlp_jacobian_gram_couples_distinct_tie_groups(self) -> None:
        """The MLP does NOT preserve tie-group orthogonality, and that is expected.

        Within a group the Jacobians coincide exactly, because identical
        features give identical activations. Across groups they do not become
        orthogonal: the output bias has the same derivative at every position,
        and the hidden activations of different groups are not orthogonal. This
        is why the calibration gate is stated against the linear head; see the
        amendment in `docs/STAGE_B_MLP_PROTOCOL.md`.
        """

        head = MLPHead(feature_dim=len(GROUPS), n_actions=N_ACTIONS, hidden=8, seed=2)
        rng = np.random.default_rng(1)
        head.W2 = rng.normal(size=head.W2.shape)
        jac = head.position_jacobians(FEATURES)
        similarity = np.clip(
            (jac / np.linalg.norm(jac, axis=1, keepdims=True))
            @ (jac / np.linalg.norm(jac, axis=1, keepdims=True)).T,
            0.0,
            None,
        )
        for members in GROUPS:
            for other in members[1:]:
                self.assertAlmostEqual(
                    similarity[members[0], other], 1.0, places=12
                )
        cross = [
            similarity[a[0], b[0]]
            for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:]
        ]
        self.assertGreater(max(cross), 0.1, "expected non-trivial cross-group coupling")


class MLPAlgorithmTest(unittest.TestCase):
    def _environment(self):
        return StructuredSequenceMDP.from_sequences(
            horizon=HORIZON, n_actions=N_ACTIONS,
            critical_positions=[0, 2, 5], target_actions=[0, 1, 2],
            minimum_matches=2,
        )

    def test_initial_policy_is_exactly_uniform(self) -> None:
        for cls, extra in (
            (MLPProjectedGroupOMD, {}),
            (MLPReliabilityWeightedProjectionOMD,
             {**RELIABILITY, "projection_lambda": 3.0}),
        ):
            with self.subTest(cls=cls.__name__):
                algorithm = cls(**BASE, **extra, mlp_hidden=8, mlp_seed=1)
                np.testing.assert_allclose(
                    algorithm.policy, 1.0 / N_ACTIONS, atol=1e-15
                )

    def test_updates_move_the_policy_and_stay_normalized(self) -> None:
        environment = self._environment()
        rng = np.random.default_rng(7)
        algorithm = MLPProjectedGroupOMD(**BASE, mlp_hidden=8, mlp_seed=1)
        for _ in range(5):
            trajectories = environment.sample(algorithm.policy, 32, rng)
            algorithm.update(trajectories, environment.batch_rewards(trajectories))
        policy = algorithm.policy
        np.testing.assert_allclose(policy.sum(axis=1), 1.0, atol=1e-12)
        self.assertGreater(float(np.abs(policy - 1.0 / N_ACTIONS).max()), 1e-6)

    def test_tied_positions_keep_identical_policies(self) -> None:
        """Shared parameters plus identical features must give identical rows.

        This is the property that makes tie-groups meaningful; if the MLP broke
        it, alpha would be measuring nothing.
        """

        environment = self._environment()
        rng = np.random.default_rng(9)
        algorithm = MLPReliabilityWeightedProjectionOMD(
            **BASE, **RELIABILITY, projection_lambda=3.0, mlp_hidden=8, mlp_seed=4
        )
        for _ in range(6):
            trajectories = environment.sample(algorithm.policy, 32, rng)
            algorithm.update(trajectories, environment.batch_rewards(trajectories))
        policy = algorithm.policy
        for members in GROUPS:
            for other in members[1:]:
                np.testing.assert_allclose(
                    policy[members[0]], policy[other], atol=1e-12
                )

    def test_mixture_target_is_inherited_unchanged(self) -> None:
        """Only the parameterization may differ from the linear RWP-OMD."""

        from algorithms import ReliabilityWeightedProjectionOMD

        kwargs = {**BASE, **RELIABILITY, "projection_lambda": 3.0}
        linear = ReliabilityWeightedProjectionOMD(**kwargs)
        nonlinear = MLPReliabilityWeightedProjectionOMD(
            **kwargs, mlp_hidden=8, mlp_seed=1
        )
        rng = np.random.default_rng(2)
        scores = rng.normal(size=(HORIZON, N_ACTIONS))
        reliability = rng.uniform(0.2, 1.0, size=HORIZON)
        for a, b in zip(
            linear.mixture_target(scores, reliability),
            nonlinear.mixture_target(scores, reliability),
        ):
            np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
