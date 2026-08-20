"""Checks for the Stage B-2 estimator menu and decision rules."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_stage_b2 import (
    discordant_pairs,
    run_calibration,
    run_confirmation,
    run_selection,
)
from experiments.geometry_v2_scenarios import structure
from experiments.geometry_v3_family import forbidden_profiles, profile_key
from experiments.stage_b2_estimators import (
    ESTIMATORS,
    REFUTED_ESTIMATOR,
    alpha_from_jacobians,
)

REPO = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    return json.loads((REPO / "configs" / name).read_text(encoding="utf-8"))


class MenuTest(unittest.TestCase):
    def test_menu_is_closed_and_contains_the_refuted_control(self) -> None:
        self.assertIn(REFUTED_ESTIMATOR, ESTIMATORS)
        self.assertEqual(len(ESTIMATORS), 6)

    def test_every_transform_preserves_position_count(self) -> None:
        rng = np.random.default_rng(0)
        jac = rng.normal(size=(7, 3 * 12))
        context = {"head": None, "n_actions": 3,
                   "policy": np.full((7, 3), 1 / 3)}
        for name, transform in ESTIMATORS.items():
            with self.subTest(name=name):
                out = transform(jac, context)
                self.assertEqual(out.shape[0], 7)

    def test_transforms_are_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        jac = rng.normal(size=(5, 3 * 9))
        context = {"head": None, "n_actions": 3,
                   "policy": np.full((5, 3), 1 / 3)}
        for name, transform in ESTIMATORS.items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    transform(jac, context), transform(jac, context)
                )

    def test_action_difference_removes_the_constant_action_shift(self) -> None:
        """The softmax is invariant to it, so the estimator should be too."""

        rng = np.random.default_rng(2)
        jac = rng.normal(size=(6, 3 * 8))
        context = {"head": None, "n_actions": 3,
                   "policy": np.full((6, 3), 1 / 3)}
        shifted = jac.reshape(6, 3, 8) + rng.normal(size=(6, 1, 8))
        np.testing.assert_allclose(
            ESTIMATORS["action_difference"](jac, context),
            ESTIMATORS["action_difference"](shifted.reshape(6, -1), context),
            atol=1e-12,
        )

    def test_alpha_is_one_when_all_positions_are_identical(self) -> None:
        """Complete aliasing: one tie-group, balanced signs, alpha must be 1."""

        jac = np.tile(np.arange(1.0, 10.0), (4, 1))
        signs = np.array([1.0, 1.0, -1.0, -1.0])
        self.assertAlmostEqual(alpha_from_jacobians(jac, signs), 1.0, places=12)


class DiscordantPairsTest(unittest.TestCase):
    def test_ties_in_the_truth_impose_no_constraint(self) -> None:
        truth = np.array([1.0, 1.0, 2.0])
        # the two tied entries are ordered arbitrarily; that must not count
        self.assertEqual(discordant_pairs(np.array([9.0, 3.0, 20.0]), truth), 0)

    def test_a_real_inversion_is_counted(self) -> None:
        truth = np.array([1.0, 2.0])
        self.assertEqual(discordant_pairs(np.array([5.0, 4.0]), truth), 1)

    def test_perfectly_monotone_scores_zero(self) -> None:
        truth = np.arange(6.0)
        self.assertEqual(discordant_pairs(np.exp(truth), truth), 0)


class CalibrationTest(unittest.TestCase):
    def test_gate_passes_and_every_variant_is_admissible(self) -> None:
        result = run_calibration(load("stage_b_mlp.json"))
        self.assertEqual(result["outcome"], "PASS")
        for name, entry in result["per_estimator"].items():
            with self.subTest(name=name):
                self.assertEqual(entry["discordant_non_tied_pairs"], 0)
                self.assertTrue(entry["admissible"])

    def test_the_tie_structure_that_motivated_the_amendment_is_real(self) -> None:
        """If ties vanish, the original Spearman gate would have been fine."""

        result = run_calibration(load("stage_b_mlp.json"))
        self.assertLess(result["distinct_discrete_values"], result["n_instances"])
        for entry in result["per_estimator"].values():
            if entry["exact_reproduction"]:
                # exact reproduction still scores below 1.0 under Spearman
                self.assertLess(entry["spearman_for_reference"], 1.0)


class ConfirmationFamilyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load("stage_b2.json")

    def test_family_is_fresh_against_every_prior_protocol(self) -> None:
        banned = set()
        for name in ("geometry_v1.json", "geometry_v2.json",
                     "geometry_v3.json", "stage_b_mlp.json"):
            banned |= forbidden_profiles([load(name)])
        for scenario in self.config["scenarios"]:
            if scenario["feature_noise"] != 0.0:
                continue
            with self.subTest(name=scenario["name"]):
                counts = tuple(tuple(p) for p in structure(scenario)["group_counts"])
                self.assertNotIn(profile_key(counts), banned)

    def test_seeds_are_disjoint_from_stage_b(self) -> None:
        stage_b = load("stage_b_mlp.json")["training"]
        used = set(stage_b["discovery_seeds"]) | set(stage_b["confirmation_seeds"])
        for scenario in self.config["scenarios"]:
            self.assertEqual(set(scenario["seeds"]) & used, set())

    def test_stage_b_confirmation_instances_are_not_reused(self) -> None:
        """They were consumed by Stage B; reusing them is a second look."""

        stage_b_names = set(load("stage_b_mlp.json")["split"]["confirmation"])
        new_names = {s["name"] for s in self.config["scenarios"]}
        self.assertEqual(stage_b_names & new_names, set())


class DecisionRuleTest(unittest.TestCase):
    def _confirm(self, alphas, deltas, monkeypatched):
        import experiments.evaluate_stage_b2 as module

        names = [f"i{i}" for i in range(len(alphas))]
        config = {
            "scenarios": [{"name": n, "seeds": [0]} for n in names],
            "methods": {"uniform_eta075000": {}},
            "training": {"evaluation_interval": 5},
            "decision_rule": {
                "candidate_method": "rwp", "grid_prefix": "uniform_eta",
                "max_dropped_seeds": 2, "expected_sign": -1,
                "confirmation_magnitude_bound": 0.50,
                "bootstrap_resamples": 300, "bootstrap_seed": 20260824,
            },
        }
        original_metric = module.scenario_metric
        original_alphas = module._alphas
        module.scenario_metric = lambda s, *a, **k: {
            "usable": True, "delta_corrected": float(deltas[names.index(s)])
        }
        module._alphas = lambda c, n, b, **k: {
            name: {monkeypatched: float(alphas[i])} for i, name in enumerate(n)
        }
        try:
            summary = {n: n for n in names}
            return module.run_confirmation(config, summary, monkeypatched)
        finally:
            module.scenario_metric = original_metric
            module._alphas = original_alphas

    def test_strong_negative_confirms(self) -> None:
        a = np.linspace(0.0, 1.0, 14)
        result = self._confirm(a, -0.05 * a + 0.06, "fisher")
        self.assertEqual(result["outcome"], "CONFIRMED")
        self.assertIn("reinstated", result["consequence"])

    def test_positive_direction_does_not_confirm(self) -> None:
        a = np.linspace(0.0, 1.0, 14)
        result = self._confirm(a, 0.05 * a, "fisher")
        self.assertFalse(result["sign_matches_expected"])
        self.assertEqual(result["outcome"], "NOT_CONFIRMED")
        self.assertIn("PROHIBITED", result["consequence"])


if __name__ == "__main__":
    unittest.main()
