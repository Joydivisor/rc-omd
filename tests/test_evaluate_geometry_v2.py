"""Checks for the geometry-v2 evaluator.

Several of these assert protocol requirements rather than ordinary code
behaviour. In particular `test_complete_aliasing_ratio_is_exactly_one` and its
companion assert that the geometric interpolation of the secondary read-out is
load-bearing: the linear alternative introduces a systematic bias that the
negative control would then report as a violation.
"""

from __future__ import annotations

import unittest

import numpy as np

from experiments.evaluate_geometry_v2 import (
    _T_CRITICAL_95,
    AUC_KEY,
    CRITICAL_KEY,
    DISTRACTOR_KEY,
    decide,
    evaluate_scenario,
    match_frontier,
    t_ci_95,
)

SEEDS = list(range(20))


def method(auc, distractor, critical, seeds=None):
    return {
        "seeds": list(SEEDS if seeds is None else seeds),
        AUC_KEY: list(map(float, auc)),
        DISTRACTOR_KEY: list(map(float, distractor)),
        CRITICAL_KEY: list(map(float, critical)),
    }


def grid_points(names_to_arrays):
    return [
        {
            "name": name,
            AUC_KEY: np.asarray(auc, dtype=float),
            DISTRACTOR_KEY: np.asarray(distractor, dtype=float),
            CRITICAL_KEY: np.asarray(critical, dtype=float),
        }
        for name, (auc, distractor, critical) in names_to_arrays.items()
    ]


class InterpolationTest(unittest.TestCase):
    def test_log_kl_interpolation_matches_hand_computation(self) -> None:
        grid = grid_points(
            {
                "uniform_a": ([0.10], [1.0], [1.0]),
                "uniform_b": ([0.30], [100.0], [100.0]),
            }
        )
        # target = 10 sits at log-midpoint of [1, 100]
        result = match_frontier(
            grid, np.array([10.0]), np.array([0.25]), DISTRACTOR_KEY, AUC_KEY
        )
        self.assertAlmostEqual(float(result["reference"][0]), 0.20, places=12)
        self.assertAlmostEqual(float(result["differences"][0]), 0.05, places=12)

    def test_bracket_chosen_by_rank_not_grid_order(self) -> None:
        """A locally non-monotone KL(eta) curve must still resolve.

        `uniform_b` is listed between the others but has the largest KL, so
        adjacency-based bracketing would pick the wrong pair.
        """

        grid = grid_points(
            {
                "uniform_a": ([0.10], [1.0], [1.0]),
                "uniform_b": ([0.50], [100.0], [100.0]),
                "uniform_c": ([0.30], [4.0], [4.0]),
            }
        )
        result = match_frontier(
            grid, np.array([2.0]), np.array([0.0]), DISTRACTOR_KEY, AUC_KEY
        )
        # Bracket must be [1.0, 4.0] -> AUC between 0.10 and 0.30, not 0.50.
        self.assertGreater(float(result["reference"][0]), 0.10)
        self.assertLess(float(result["reference"][0]), 0.30)

    def test_duplicate_kl_falls_back_to_lower_point(self) -> None:
        grid = grid_points(
            {
                "uniform_a": ([0.10], [5.0], [5.0]),
                "uniform_b": ([0.30], [5.0], [5.0]),
            }
        )
        result = match_frontier(
            grid, np.array([5.0]), np.array([0.40]), DISTRACTOR_KEY, AUC_KEY
        )
        self.assertEqual(result["dropped_seeds"], 0)
        self.assertTrue(np.isfinite(result["reference"][0]))
        self.assertIn(float(result["reference"][0]), (0.10, 0.30))

    def test_out_of_bracket_seeds_are_dropped_and_counted(self) -> None:
        grid = grid_points({"uniform_a": ([0.1, 0.1], [10.0, 10.0], [1.0, 1.0])})
        # first seed below the grid's minimum, second above its maximum
        result = match_frontier(
            grid, np.array([1.0, 100.0]), np.array([0.5, 0.5]),
            DISTRACTOR_KEY, AUC_KEY,
        )
        self.assertEqual(result["dropped_seeds"], 2)
        self.assertEqual(result["dropped_index"], [0, 1])
        self.assertEqual(result["differences"].size, 0)


class SecondaryMetricTest(unittest.TestCase):
    """Control 1 is exact only under geometric interpolation of the read-out."""

    def _complete_aliasing_methods(self):
        rng = np.random.default_rng(11)
        methods = {}
        for index, eta in enumerate([0.25, 0.5, 1.0, 2.0]):
            shared = eta * (1.0 + 0.2 * rng.random(len(SEEDS)))
            methods[f"uniform_eta{index}"] = method(
                0.5 + 0.1 * index + 0.01 * rng.random(len(SEEDS)), shared, shared
            )
        candidate_shared = 0.7 * (1.0 + 0.2 * rng.random(len(SEEDS)))
        methods["rwp"] = method(
            0.6 + 0.01 * rng.random(len(SEEDS)), candidate_shared, candidate_shared
        )
        return methods

    def test_complete_aliasing_ratio_is_exactly_one(self) -> None:
        result = evaluate_scenario(
            "complete", self._complete_aliasing_methods(), "rwp", "uniform_eta", 2
        )
        allocation = result["secondary"]["allocation"]
        self.assertIsNotNone(allocation)
        self.assertLessEqual(allocation["max_abs_deviation_from_one"], 1e-9)
        self.assertAlmostEqual(allocation["ratio_mean"], 1.0, places=12)

    def test_linear_readout_would_break_the_invariant(self) -> None:
        """Justifies the geometric rule: the linear alternative is biased.

        Under complete aliasing the read-out equals the matching axis, so any
        correct interpolation must return the target exactly.
        """

        methods = self._complete_aliasing_methods()
        grid = grid_points(
            {
                name: (m[AUC_KEY], m[DISTRACTOR_KEY], m[CRITICAL_KEY])
                for name, m in methods.items()
                if name.startswith("uniform_eta")
            }
        )
        candidate = methods["rwp"]
        target = np.asarray(candidate[CRITICAL_KEY], dtype=float)
        readout = np.asarray(candidate[DISTRACTOR_KEY], dtype=float)

        geometric = match_frontier(
            grid, target, readout, CRITICAL_KEY, DISTRACTOR_KEY, log_readout=True
        )
        linear = match_frontier(
            grid, target, readout, CRITICAL_KEY, DISTRACTOR_KEY, log_readout=False
        )
        geometric_error = np.abs(geometric["reference"] / readout - 1.0).max()
        linear_error = np.abs(linear["reference"] / readout - 1.0).max()
        self.assertLessEqual(geometric_error, 1e-12)
        self.assertGreater(linear_error, 1e-6)


class VerdictTest(unittest.TestCase):
    def _scenario(self, candidate_auc_offset: float, dropped: bool = False):
        rng = np.random.default_rng(3)
        methods = {}
        for index, base in enumerate([0.20, 0.40, 0.60, 0.80]):
            kl = (index + 1) * (1.0 + 0.05 * rng.random(len(SEEDS)))
            methods[f"uniform_eta{index}"] = method(
                base + 0.001 * rng.random(len(SEEDS)), kl, kl * 2.0
            )
        anchor = 2.5 * (1.0 + 0.05 * rng.random(len(SEEDS)))
        if dropped:
            anchor = anchor * 1e-6
        methods["rwp"] = method(
            0.50 + candidate_auc_offset + 0.001 * rng.random(len(SEEDS)),
            anchor,
            anchor * 2.0,
        )
        return methods

    def test_positive_verdict(self) -> None:
        result = evaluate_scenario(
            "s", self._scenario(+0.20), "rwp", "uniform_eta", 2
        )
        self.assertEqual(result["primary"]["verdict"], "positive")
        self.assertGreater(result["primary"]["delta_frontier"]["ci_lower"], 0.0)

    def test_negative_verdict(self) -> None:
        result = evaluate_scenario(
            "s", self._scenario(-0.20), "rwp", "uniform_eta", 2
        )
        self.assertEqual(result["primary"]["verdict"], "negative")

    def test_null_verdict_when_interval_spans_zero(self) -> None:
        rng = np.random.default_rng(5)
        methods = {}
        for index in range(4):
            kl = (index + 1) * (1.0 + 0.05 * rng.random(len(SEEDS)))
            methods[f"uniform_eta{index}"] = method(
                0.20 * (index + 1) + 0.10 * rng.standard_normal(len(SEEDS)),
                kl,
                kl * 2.0,
            )
        anchor = 2.5 * (1.0 + 0.05 * rng.random(len(SEEDS)))
        methods["rwp"] = method(
            0.50 + 0.10 * rng.standard_normal(len(SEEDS)), anchor, anchor * 2.0
        )
        result = evaluate_scenario("s", methods, "rwp", "uniform_eta", 2)
        self.assertEqual(result["primary"]["verdict"], "null")

    def test_excess_dropped_seeds_yield_inconclusive(self) -> None:
        result = evaluate_scenario(
            "s", self._scenario(0.0, dropped=True), "rwp", "uniform_eta", 2
        )
        self.assertEqual(result["primary"]["verdict"], "inconclusive")
        self.assertIsNone(result["primary"]["delta_frontier"])
        self.assertGreater(result["primary"]["dropped_seeds"], 2)

    def test_seed_mismatch_is_rejected(self) -> None:
        methods = self._scenario(0.0)
        methods["uniform_eta0"]["seeds"] = list(range(100, 120))
        with self.assertRaises(ValueError):
            evaluate_scenario("s", methods, "rwp", "uniform_eta", 2)


class StatisticsTest(unittest.TestCase):
    def test_t_critical_table_matches_geometry_v1(self) -> None:
        """The two protocols must compute identical intervals."""

        from experiments.evaluate_geometry_protocol import (
            _T_CRITICAL_95 as V1_TABLE,
        )

        self.assertEqual(_T_CRITICAL_95, V1_TABLE)

    def test_unsupported_degrees_of_freedom_rejected(self) -> None:
        with self.assertRaises(ValueError):
            t_ci_95(np.zeros(5))

    def test_interval_is_symmetric_about_the_mean(self) -> None:
        values = np.linspace(-1.0, 1.0, 20)
        interval = t_ci_95(values)
        self.assertAlmostEqual(
            interval["ci_upper"] - interval["mean"],
            interval["mean"] - interval["ci_lower"],
            places=12,
        )


class DecisionTest(unittest.TestCase):
    SCENARIOS = [
        "geom_v2_a3_pure_ge1",
        "geom_v2_a3_pure0_homog",
        "geom_v2_a3_pure0_hetero",
        "separable_shared_features",
        "geom_holdout_a",
        "geom_holdout_b",
        "partial_feature_aliasing",
        "complete_feature_aliasing_negative_control",
    ]

    def _config(self):
        base = {
            "geom_v2_a3_pure_ge1": "positive",
            "geom_v2_a3_pure0_homog": "null",
            "separable_shared_features": "positive",
            "geom_holdout_a": "positive",
            "geom_holdout_b": "positive",
            "partial_feature_aliasing": "null",
            "complete_feature_aliasing_negative_control": "null_or_negative",
        }
        return {
            "decision_rule": {
                "invariant_scenario": "complete_feature_aliasing_negative_control",
                "invariant_tolerance": 1e-9,
                "discriminating_scenario": "geom_v2_a3_pure0_hetero",
                "no_spurious_pass_scenarios": [
                    "geom_v2_a3_pure0_homog",
                    "partial_feature_aliasing",
                ],
                "predictions_h_pure": {**base, "geom_v2_a3_pure0_hetero": "null"},
                "predictions_h_hetero": {
                    **base,
                    "geom_v2_a3_pure0_hetero": "positive",
                },
            }
        }

    def _results(self, verdicts, invariant_deviation=0.0):
        results = {}
        for name in self.SCENARIOS:
            allocation = None
            if name == "complete_feature_aliasing_negative_control":
                allocation = {
                    "max_abs_deviation_from_one": invariant_deviation,
                    "ratio_mean": 1.0,
                    "ratio_ci": [1.0, 1.0],
                    "log_ratio": {},
                }
            results[name] = {
                "scenario": name,
                "primary": {"verdict": verdicts[name], "delta_frontier": None,
                            "dropped_seeds": 0, "dropped_index": []},
                "secondary": {"allocation": allocation, "dropped_seeds": 0},
            }
        return results

    def _verdicts(self, **overrides):
        base = {
            "geom_v2_a3_pure_ge1": "positive",
            "geom_v2_a3_pure0_homog": "null",
            "geom_v2_a3_pure0_hetero": "null",
            "separable_shared_features": "positive",
            "geom_holdout_a": "positive",
            "geom_holdout_b": "positive",
            "partial_feature_aliasing": "null",
            "complete_feature_aliasing_negative_control": "negative",
        }
        base.update(overrides)
        return base

    def test_go_when_discriminator_is_null(self) -> None:
        decision = decide(self._config(), self._results(self._verdicts()))
        self.assertEqual(decision["outcome"], "GO_H_PURE_CONFIRMED")
        self.assertTrue(decision["hypothesis_h_pure_satisfied"])
        self.assertFalse(decision["hypothesis_h_hetero_satisfied"])

    def test_revise_when_discriminator_is_positive(self) -> None:
        verdicts = self._verdicts(geom_v2_a3_pure0_hetero="positive")
        decision = decide(self._config(), self._results(verdicts))
        self.assertEqual(decision["outcome"], "REVISE_H_HETERO_CONFIRMED")
        self.assertTrue(decision["hypothesis_h_hetero_satisfied"])

    def test_no_go_when_positive_arm_fails(self) -> None:
        verdicts = self._verdicts(geom_v2_a3_pure_ge1="null")
        decision = decide(self._config(), self._results(verdicts))
        self.assertEqual(decision["outcome"], "NO-GO_LAW_REFUTED")

    def test_control_two_flags_spurious_pass(self) -> None:
        verdicts = self._verdicts(partial_feature_aliasing="positive")
        decision = decide(self._config(), self._results(verdicts))
        self.assertFalse(decision["control_2_no_spurious_pass"]["holds"])
        self.assertEqual(decision["outcome"], "NO-GO_LAW_REFUTED")

    def test_invariant_violation_halts(self) -> None:
        decision = decide(
            self._config(),
            self._results(self._verdicts(), invariant_deviation=1e-3),
        )
        self.assertEqual(decision["outcome"], "HALT_INVARIANT_VIOLATED")

    def test_inconclusive_scenario_blocks_a_decision(self) -> None:
        verdicts = self._verdicts(geom_holdout_a="inconclusive")
        decision = decide(self._config(), self._results(verdicts))
        self.assertEqual(decision["outcome"], "INCONCLUSIVE")

    def test_complete_aliasing_may_be_negative_but_not_positive(self) -> None:
        null_case = decide(
            self._config(),
            self._results(
                self._verdicts(complete_feature_aliasing_negative_control="null")
            ),
        )
        self.assertEqual(null_case["outcome"], "GO_H_PURE_CONFIRMED")
        positive_case = decide(
            self._config(),
            self._results(
                self._verdicts(complete_feature_aliasing_negative_control="positive")
            ),
        )
        self.assertEqual(positive_case["outcome"], "NO-GO_LAW_REFUTED")


if __name__ == "__main__":
    unittest.main()
