"""Checks for the geometry-v3 family, split, and evaluator.

The split assertions matter more than they look: the protocol's entire defence
against repeating the geometry-v2 mistake is that discovery and confirmation are
disjoint and fixed before execution. If those properties stop holding, the
confirmation result means nothing.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_geometry_v3 import (
    bootstrap_rho_interval,
    rankdata,
    run_confirmation,
    run_discovery,
    spearman,
)
from experiments.geometry_v2_scenarios import is_realizable, structure
from experiments.geometry_v3_family import (
    DISCOVERY_SIZE,
    FAMILY_SIZE,
    PREDICTORS,
    REFUTED_PREDICTORS,
    build_scenario,
    forbidden_profiles,
    profile_key,
    split_family,
)

REPO = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    return json.loads((REPO / "configs" / name).read_text(encoding="utf-8"))


class FamilyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load("geometry_v3.json")
        self.control = self.config["control_scenario"]
        self.family = [
            s for s in self.config["scenarios"] if s["name"] != self.control
        ]

    def test_family_size(self) -> None:
        self.assertEqual(len(self.family), FAMILY_SIZE)

    def test_split_is_disjoint_and_covers_the_family(self) -> None:
        split = self.config["split"]
        discovery, confirmation = set(split["discovery"]), set(split["confirmation"])
        self.assertEqual(len(discovery), DISCOVERY_SIZE)
        self.assertEqual(len(confirmation), FAMILY_SIZE - DISCOVERY_SIZE)
        self.assertEqual(discovery & confirmation, set())
        self.assertEqual(discovery | confirmation,
                         {s["name"] for s in self.family})

    def test_split_is_reproducible_from_the_committed_seed(self) -> None:
        names = [s["name"] for s in self.family]
        self.assertEqual(split_family(names), self.config["split"])

    def test_control_is_outside_the_family_and_the_split(self) -> None:
        split = self.config["split"]
        self.assertNotIn(self.control, split["discovery"])
        self.assertNotIn(self.control, split["confirmation"])
        self.assertNotIn(self.control, {s["name"] for s in self.family})

    def test_control_is_complete_aliasing(self) -> None:
        scenario = next(
            s for s in self.config["scenarios"] if s["name"] == self.control
        )
        self.assertEqual(structure(scenario)["alpha"], 1.0)

    def test_no_profile_reused_from_earlier_protocols(self) -> None:
        banned = forbidden_profiles([load("geometry_v1.json"), load("geometry_v2.json")])
        for scenario in self.family:
            with self.subTest(name=scenario["name"]):
                counts = tuple(tuple(p) for p in structure(scenario)["group_counts"])
                self.assertNotIn(profile_key(counts), banned)

    def test_family_profiles_are_distinct(self) -> None:
        keys = [
            profile_key(tuple(tuple(p) for p in structure(s)["group_counts"]))
            for s in self.family
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_member_is_realizable(self) -> None:
        for scenario in self.config["scenarios"]:
            with self.subTest(name=scenario["name"]):
                self.assertTrue(is_realizable(scenario))

    def test_every_member_met_the_learnability_threshold(self) -> None:
        threshold = self.config["generation"]["learnability_threshold"]
        for name, gain in self.config["generation"]["baseline_gain"].items():
            with self.subTest(name=name):
                self.assertGreaterEqual(gain, threshold)

    def test_predictor_menu_is_closed_and_total(self) -> None:
        for scenario in self.config["scenarios"]:
            values = self.config["predictors"][scenario["name"]]
            self.assertEqual(set(values), set(PREDICTORS))
            for key, value in values.items():
                with self.subTest(name=scenario["name"], predictor=key):
                    self.assertTrue(np.isfinite(value))

    def test_refuted_predictors_remain_in_the_menu(self) -> None:
        """They are the overfitting check; removing them defeats it."""

        for name in REFUTED_PREDICTORS:
            self.assertIn(name, PREDICTORS)

    def test_generation_rule_is_deterministic(self) -> None:
        profile = ((0, 2), (2, 1), (1, 1))
        self.assertEqual(
            json.dumps(build_scenario("x", profile)),
            json.dumps(build_scenario("x", profile)),
        )


class CorrelationTest(unittest.TestCase):
    def test_rankdata_averages_ties(self) -> None:
        np.testing.assert_allclose(
            rankdata(np.array([10.0, 20.0, 20.0, 30.0])), [1.0, 2.5, 2.5, 4.0]
        )

    def test_spearman_perfect_monotone(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(spearman(x, np.exp(x)), 1.0, places=12)
        self.assertAlmostEqual(spearman(x, -np.exp(x)), -1.0, places=12)

    def test_spearman_constant_input_is_zero(self) -> None:
        self.assertEqual(spearman(np.ones(5), np.arange(5.0)), 0.0)

    def test_bootstrap_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        y = x + 0.3 * rng.normal(size=30)
        interval = bootstrap_rho_interval(x, y, 500, 1234)
        self.assertLessEqual(interval["ci_lower"], interval["rho"])
        self.assertGreaterEqual(interval["ci_upper"], interval["rho"])

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self) -> None:
        x = np.arange(20.0)
        y = x**2
        a = bootstrap_rho_interval(x, y, 200, 7)
        b = bootstrap_rho_interval(x, y, 200, 7)
        self.assertEqual(a, b)


class SyntheticDecisionTest(unittest.TestCase):
    """Drive the decision paths without a real sweep."""

    SEEDS = list(range(20))

    def _methods(self, delta: float, complete: bool = False):
        rng = np.random.default_rng(4)
        methods = {}
        for index, eta in enumerate([0.25, 0.5, 1.0, 2.0]):
            kl = eta * (1.0 + 0.1 * rng.random(len(self.SEEDS)))
            methods[f"uniform_eta{index}"] = {
                "seeds": list(self.SEEDS),
                "success_auc_per_seed": list(0.2 * (index + 1)
                                             + 0.001 * rng.random(len(self.SEEDS))),
                "cumulative_distractor_kl_per_seed": list(kl),
                "cumulative_critical_kl_per_seed": list(kl if complete else kl * 2.0),
            }
        kl = 0.75 * (1.0 + 0.1 * rng.random(len(self.SEEDS)))
        methods["rwp_eta125_lam3"] = {
            "seeds": list(self.SEEDS),
            "success_auc_per_seed": list(0.4 + delta
                                         + 0.001 * rng.random(len(self.SEEDS))),
            "cumulative_distractor_kl_per_seed": list(kl),
            "cumulative_critical_kl_per_seed": list(kl if complete else kl * 2.0),
        }
        return methods

    def _config(self, n_discovery=6, n_confirmation=5):
        names_d = [f"d{i}" for i in range(n_discovery)]
        names_c = [f"c{i}" for i in range(n_confirmation)]
        predictors = {}
        for index, name in enumerate(names_d + names_c):
            predictors[name] = {"good": float(index), "flat": 1.0,
                                "pure_crit": float(index % 2)}
        predictors["ctrl"] = {"good": 0.0, "flat": 1.0, "pure_crit": 0.0}
        return {
            "protocol_id": "geometry-v3-2026-08-19",
            "split": {"discovery": names_d, "confirmation": names_c},
            "control_scenario": "ctrl",
            "predictors": predictors,
            "decision_rule": {
                "candidate_method": "rwp_eta125_lam3",
                "grid_prefix": "uniform_eta",
                "max_dropped_seeds": 2,
                "selection_threshold_abs_rho": 0.70,
                "confirmation_rho_lower_bound": 0.50,
                "bootstrap_resamples": 300,
                "bootstrap_seed": 20260820,
                "invariant_tolerance": 1e-9,
            },
        }

    def _summary(self, config, deltas):
        summary = {
            name: self._methods(deltas[index])
            for index, name in enumerate(
                config["split"]["discovery"] + config["split"]["confirmation"]
            )
        }
        summary["ctrl"] = self._methods(0.0, complete=True)
        return summary

    def test_control_invariant_is_exact_and_selection_runs(self) -> None:
        config = self._config()
        deltas = [0.01 * i for i in range(11)]
        result = run_discovery(config, self._summary(config, deltas))
        self.assertTrue(result["control"]["invariant_holds"])
        self.assertLessEqual(
            result["control"]["allocation_max_abs_deviation_from_one"], 1e-9
        )
        self.assertEqual(result["outcome"], "SELECTED")
        self.assertEqual(result["selected"]["predictor"], "good")

    def test_no_predictor_found_when_response_is_unrelated(self) -> None:
        config = self._config()
        # deltas deliberately non-monotone in the predictor index
        deltas = [0.05, 0.01, 0.04, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = run_discovery(config, self._summary(config, deltas))
        self.assertIn(result["outcome"], ("NO_PREDICTOR_FOUND", "SELECTED"))
        if result["outcome"] == "NO_PREDICTOR_FOUND":
            self.assertIsNone(result["selected"])

    def test_discovery_never_touches_confirmation_scenarios(self) -> None:
        config = self._config()
        deltas = [0.01 * i for i in range(11)]
        result = run_discovery(config, self._summary(config, deltas))
        self.assertEqual(set(result["metrics"]), set(config["split"]["discovery"]))
        for name in config["split"]["confirmation"]:
            self.assertNotIn(name, result["metrics"])

    def test_confirmation_reports_an_interval_and_a_verdict(self) -> None:
        config = self._config()
        deltas = [0.01 * i for i in range(11)]
        result = run_confirmation(config, self._summary(config, deltas), "good", 1)
        self.assertIn(result["outcome"], ("CONFIRMED", "NOT_CONFIRMED"))
        self.assertEqual(set(result["metrics"]), set(config["split"]["confirmation"]))
        self.assertIn("ci_lower", result["spearman"])

    def test_a_negative_predictor_can_confirm(self) -> None:
        """Regression for the amended criterion.

        The selection rule ranks by absolute correlation, so a negative
        predictor is reachable. Under the original wording -- a bound stated
        only as "lower bound > 0.50" -- such a predictor could never confirm
        however strong it was. The bound is on magnitude in the discovery
        direction.
        """

        config = self._config()
        # response decreasing in the predictor index => rho is strongly negative
        deltas = [0.10 - 0.01 * i for i in range(11)]
        summary = self._summary(config, deltas)
        negative = run_confirmation(config, summary, "good", -1)
        self.assertLess(negative["spearman"]["rho"], 0.0)
        self.assertTrue(negative["sign_matches_discovery"])
        self.assertEqual(negative["outcome"], "CONFIRMED")

        # The same data must NOT confirm if discovery had said positive.
        wrong_sign = run_confirmation(config, summary, "good", 1)
        self.assertFalse(wrong_sign["sign_matches_discovery"])
        self.assertEqual(wrong_sign["outcome"], "NOT_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
