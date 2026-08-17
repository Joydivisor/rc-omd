"""Numerical invariants for RWP-OMD.

These are the ten checks required by `docs/PARAMETER_SPACE_GEOMETRY_DESIGN.md`
Section 13, and by the geometry-v1 protocol's freeze stage 2. Several of them
are not ordinary regression tests but assertions of properties the design
document treats as theorems; where that is the case the docstring says so.
"""

from __future__ import annotations

import unittest

import numpy as np

from algorithms import ProjectedGroupOMD, ReliabilityWeightedProjectionOMD
from environments import StructuredSequenceMDP


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(p: np.ndarray, pi: np.ndarray) -> float:
    return float(-(p * np.log(pi)).sum())


BASE = dict(
    horizon=4,
    n_actions=3,
    step_size=1.25,
    normalize_advantages=False,
    importance_clip=20.0,
    projection_steps=60,
    projection_learning_rate=0.5,
    projection_ridge=0.0,
    projection_tolerance=1e-9,
)


def make(features: np.ndarray, **overrides: object) -> ReliabilityWeightedProjectionOMD:
    kwargs = dict(BASE)
    kwargs.update(overrides)
    horizon, n_actions = int(kwargs["horizon"]), int(kwargs["n_actions"])
    return ReliabilityWeightedProjectionOMD(
        features=features,
        initial_policy=np.full((horizon, n_actions), 1.0 / n_actions),
        **kwargs,
    )


SHARED_FEATURES = np.asarray(
    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
)  # positions 0,1 tied and 2,3 tied
EYE_FEATURES = np.eye(4)


class MixtureIdentityTest(unittest.TestCase):
    def test_aggregation_identity_holds(self) -> None:
        """Invariant 3. The two forward-KL terms must collapse to a single
        weighted cross-entropy against the arithmetic mixture, up to a constant
        that does not depend on the parametric policy."""
        rng = np.random.default_rng(0)
        lam = 2.5
        algorithm = make(EYE_FEATURES, projection_lambda=lam)
        scores = rng.normal(size=(4, 3))
        r = rng.uniform(0.1, 0.9, size=4)
        old = algorithm.policy
        target, mixture, _ = algorithm.mixture_target(scores, r)
        weights = r + lam * (1.0 - r)

        offsets = []
        for _ in range(5):
            pi = softmax(rng.normal(size=(4, 3)))
            lhs = sum(
                r[k] * cross_entropy(target[k], pi[k])
                + lam * (1.0 - r[k]) * cross_entropy(old[k], pi[k])
                for k in range(4)
            )
            rhs = sum(weights[k] * cross_entropy(mixture[k], pi[k]) for k in range(4))
            offsets.append(lhs - rhs)
        self.assertTrue(
            np.allclose(offsets, offsets[0], atol=1e-9),
            f"offset varies with pi, so the collapse is wrong: {offsets}",
        )

    def test_analytic_gradient_matches_finite_differences(self) -> None:
        """Invariant 4."""
        rng = np.random.default_rng(1)
        lam, mu = 2.5, 0.05
        features = rng.normal(size=(4, 3))
        algorithm = make(features, projection_lambda=lam, projection_ridge=mu)
        scores = rng.normal(size=(4, 3))
        r = rng.uniform(0.1, 0.9, size=4)
        _, mixture, normalized = algorithm.mixture_target(scores, r)
        theta_old = algorithm.weights.copy()

        def loss(flat: np.ndarray) -> float:
            theta = flat.reshape(3, 3)
            pi = softmax(features @ theta)
            value = sum(
                normalized[k] * cross_entropy(mixture[k], pi[k]) for k in range(4)
            )
            return value / 4.0 + 0.5 * mu * float(((theta - theta_old) ** 2).sum())

        def analytic(flat: np.ndarray) -> np.ndarray:
            theta = flat.reshape(3, 3)
            pi = softmax(features @ theta)
            grad = features.T @ (normalized[:, None] * (pi - mixture)) / 4.0
            grad += mu * (theta - theta_old)
            return grad.ravel()

        x = rng.normal(size=9) * 0.4
        numeric = np.zeros_like(x)
        h = 1e-6
        for i in range(x.size):
            up, down = x.copy(), x.copy()
            up[i] += h
            down[i] -= h
            numeric[i] = (loss(up) - loss(down)) / (2 * h)
        np.testing.assert_allclose(analytic(x), numeric, atol=1e-7)

    def test_objective_is_convex_in_theta(self) -> None:
        """Invariant 10. Convexity is what the forward-forward direction was
        chosen for; a negative eigenvalue means the objective was changed."""
        rng = np.random.default_rng(2)
        features = rng.normal(size=(4, 3))
        algorithm = make(features, projection_lambda=3.0)
        scores = rng.normal(size=(4, 3))
        r = rng.uniform(0.1, 0.9, size=4)
        _, mixture, normalized = algorithm.mixture_target(scores, r)

        def loss(flat: np.ndarray) -> float:
            pi = softmax(features @ flat.reshape(3, 3))
            return sum(normalized[k] * cross_entropy(mixture[k], pi[k]) for k in range(4))

        for _ in range(3):
            x = rng.normal(size=9) * 0.7
            n, h = x.size, 1e-4
            hessian = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    pp, pm, mp, mm = x.copy(), x.copy(), x.copy(), x.copy()
                    pp[i] += h; pp[j] += h
                    pm[i] += h; pm[j] -= h
                    mp[i] -= h; mp[j] += h
                    mm[i] -= h; mm[j] -= h
                    hessian[i, j] = (loss(pp) - loss(pm) - loss(mp) + loss(mm)) / (4 * h * h)
            hessian = (hessian + hessian.T) / 2
            self.assertGreater(np.linalg.eigvalsh(hessian).min(), -1e-5)


class DegeneracyTest(unittest.TestCase):
    def test_policy_stays_on_the_simplex(self) -> None:
        """Invariant 1."""
        rng = np.random.default_rng(3)
        algorithm = make(SHARED_FEATURES, projection_lambda=2.0)
        environment = StructuredSequenceMDP.from_sequences(
            horizon=4, n_actions=3, critical_positions=(0, 2),
            target_actions=(1, 2), minimum_matches=1,
        )
        for _ in range(15):
            trajectories = environment.sample(algorithm.policy, 32, rng)
            algorithm.update(trajectories, environment.batch_rewards(trajectories))
            policy = algorithm.policy
            np.testing.assert_allclose(policy.sum(axis=1), 1.0)
            self.assertTrue(np.all(policy > 0.0))

    def test_action_shift_invariance(self) -> None:
        """Invariant 2. Adding a constant to every action score at a position
        must not change the update, since softmax is shift-invariant."""
        rng = np.random.default_rng(4)
        scores = rng.normal(size=(4, 3))
        r = rng.uniform(0.2, 0.9, size=4)
        shift = rng.normal(size=(4, 1))

        a = make(SHARED_FEATURES, projection_lambda=2.0)
        a._reliability_weighted_projection(scores, r)
        b = make(SHARED_FEATURES, projection_lambda=2.0)
        b._reliability_weighted_projection(scores + shift, r)
        np.testing.assert_allclose(a.policy, b.policy, atol=1e-12)

    def test_zero_reliability_produces_no_movement(self) -> None:
        """Invariant 5. r = 0 everywhere means the mixture is the current
        policy, so the projection has nothing to move toward."""
        rng = np.random.default_rng(5)
        algorithm = make(SHARED_FEATURES, projection_lambda=2.0)
        before_policy = algorithm.policy.copy()
        before_weights = algorithm.weights.copy()
        stats = algorithm._reliability_weighted_projection(
            rng.normal(size=(4, 3)), np.zeros(4)
        )
        np.testing.assert_allclose(algorithm.policy, before_policy, atol=1e-12)
        np.testing.assert_allclose(algorithm.weights, before_weights, atol=1e-12)
        self.assertAlmostEqual(stats["kl_drift"], 0.0, places=12)
        self.assertAlmostEqual(stats["target_kl_total"], 0.0, places=12)

    def test_unit_reliability_recovers_projected_group_omd(self) -> None:
        """Invariant 6. At r = 1 the weights are uniform and the mixture is the
        raw OMD target, so the objective is exactly the existing unweighted
        forward-KL projection."""
        rng = np.random.default_rng(6)
        scores = rng.normal(size=(4, 3))
        candidate = make(SHARED_FEATURES, projection_lambda=2.5)
        candidate._reliability_weighted_projection(scores, np.ones(4))

        baseline = ProjectedGroupOMD(
            features=SHARED_FEATURES,
            initial_policy=np.full((4, 3), 1.0 / 3.0),
            **BASE,
        )
        baseline._apply_action_scores(scores)
        np.testing.assert_allclose(candidate.weights, baseline.weights, atol=1e-12)
        np.testing.assert_allclose(candidate.policy, baseline.policy, atol=1e-12)

    def test_one_hot_features_reproduce_the_mixture(self) -> None:
        """Invariant 7. With no coupling the projection is exact, so the update
        is the arithmetic reliability interpolation."""
        rng = np.random.default_rng(7)
        algorithm = make(
            EYE_FEATURES,
            projection_lambda=2.0,
            projection_steps=20000,
            projection_learning_rate=5.0,
        )
        scores = rng.normal(size=(4, 3))
        r = rng.uniform(0.2, 0.9, size=4)
        _, mixture, _ = algorithm.mixture_target(scores, r)
        algorithm._reliability_weighted_projection(scores, r)
        np.testing.assert_allclose(algorithm.policy, mixture, atol=1e-4)

    def test_projection_residual_is_reproducible(self) -> None:
        """Invariant 9. Determinism does not follow from convexity alone, so it
        is asserted directly."""
        rng_scores = np.random.default_rng(8).normal(size=(4, 3))
        r = np.random.default_rng(9).uniform(0.1, 0.9, size=4)
        residuals = []
        for _ in range(3):
            algorithm = make(SHARED_FEATURES, projection_lambda=2.0)
            residuals.append(
                algorithm._reliability_weighted_projection(rng_scores, r)[
                    "projection_residual_total"
                ]
            )
        self.assertEqual(len(set(residuals)), 1, f"non-reproducible: {residuals}")


class CompleteAliasingNegativeControlTest(unittest.TestCase):
    """Invariant 8. At alpha = 1 every tie-group has c(g) = d(g), so realized
    critical KL equals realized distractor KL for *any* update in this
    parametric family. The design spec proves this; a violation here is an
    implementation defect, and the protocol requires halting on it rather than
    reporting a result."""

    def test_critical_and_distractor_kl_are_exactly_equal(self) -> None:
        rng = np.random.default_rng(10)
        # positions 0,1 tied (0 critical, 1 distractor); 2,3 tied likewise.
        algorithm = make(SHARED_FEATURES, projection_lambda=3.0)
        environment = StructuredSequenceMDP.from_sequences(
            horizon=4, n_actions=3, critical_positions=(0, 2),
            target_actions=(1, 2), minimum_matches=1,
        )
        critical = np.asarray(environment.critical_positions)
        distractors = np.asarray(environment.distractor_positions)
        self.assertEqual(sorted(distractors.tolist()), [1, 3])

        total_critical = 0.0
        total_distractor = 0.0
        for _ in range(12):
            old = algorithm.policy.copy()
            trajectories = environment.sample(old, 32, rng)
            algorithm.update(trajectories, environment.batch_rewards(trajectories))
            new = algorithm.policy
            per_position = np.sum(new * (np.log(new) - np.log(old)), axis=1)
            total_critical += float(per_position[critical].sum())
            total_distractor += float(per_position[distractors].sum())

        self.assertGreater(total_critical, 0.0, "no movement, test is vacuous")
        self.assertAlmostEqual(total_critical, total_distractor, places=9)


if __name__ == "__main__":
    unittest.main()
