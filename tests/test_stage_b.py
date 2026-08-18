"""Checks for the Stage B family, alpha estimator, and decision rules."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_stage_b import run_calibration, run_confirmation
from experiments.geometry_v2_scenarios import is_realizable, structure
from experiments.geometry_v3_family import forbidden_profiles, profile_key
from experiments.run_reliability_diagnostics import build_algorithm
from experiments.stage_b_alpha import (
    alpha_from_jacobians,
    measure_alpha,
    position_jacobians,
)

REPO = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    return json.loads((REPO / "configs" / name).read_text(encoding="utf-8"))


class FamilyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load("stage_b_mlp.json")
        self.by_name = {s["name"]: s for s in self.config["scenarios"]}

    def test_every_base_appears_at_every_noise_level(self) -> None:
        bases = {s["base_scenario"] for s in self.config["scenarios"]}
        for base in bases:
            levels = sorted(
                s["feature_noise"] for s in self.config["scenarios"]
                if s["base_scenario"] == base
            )
            self.assertEqual(levels, sorted(self.config["eps_levels"]), base)

    def test_split_is_by_base_scenario_not_by_instance(self) -> None:
        """Otherwise the same structure leaks across the split via noise levels."""

        split = self.config["split"]
        discovery_bases = {
            self.by_name[n]["base_scenario"] for n in split["discovery"]
        }
        confirmation_bases = {
            self.by_name[n]["base_scenario"] for n in split["confirmation"]
        }
        self.assertEqual(discovery_bases & confirmation_bases, set())

    def test_split_is_disjoint_and_control_excluded(self) -> None:
        split = self.config["split"]
        self.assertEqual(set(split["discovery"]) & set(split["confirmation"]), set())
        for name in self.config["control_instances"]:
            self.assertNotIn(name, split["discovery"])
            self.assertNotIn(name, split["confirmation"])

    def test_discovery_and_confirmation_seeds_are_disjoint(self) -> None:
        split = self.config["split"]
        discovery = {s for n in split["discovery"] for s in self.by_name[n]["seeds"]}
        confirmation = {
            s for n in split["confirmation"] for s in self.by_name[n]["seeds"]
        }
        self.assertEqual(discovery & confirmation, set())
        self.assertEqual(discovery, set(self.config["training"]["discovery_seeds"]))
        self.assertEqual(
            confirmation, set(self.config["training"]["confirmation_seeds"])
        )

    def test_no_profile_reused_from_earlier_protocols(self) -> None:
        banned = set()
        for name in ("geometry_v1.json", "geometry_v2.json", "geometry_v3.json"):
            banned |= forbidden_profiles([load(name)])
        for scenario in self.config["scenarios"]:
            if scenario["base_scenario"] == self.config["control_base"]:
                continue
            if scenario["feature_noise"] != 0.0:
                continue
            with self.subTest(name=scenario["name"]):
                counts = tuple(tuple(p) for p in structure(scenario)["group_counts"])
                self.assertNotIn(profile_key(counts), banned)

    def test_noise_changes_features_but_not_the_task(self) -> None:
        for base in {s["base_scenario"] for s in self.config["scenarios"]}:
            instances = [
                s for s in self.config["scenarios"] if s["base_scenario"] == base
            ]
            reference = instances[0]
            for other in instances[1:]:
                with self.subTest(base=base, eps=other["feature_noise"]):
                    self.assertEqual(
                        reference["critical_positions"], other["critical_positions"]
                    )
                    self.assertEqual(
                        reference["target_actions"], other["target_actions"]
                    )
                    self.assertEqual(
                        reference["minimum_matches"], other["minimum_matches"]
                    )
                    if other["feature_noise"] > 0:
                        self.assertFalse(
                            np.allclose(reference["features"], other["features"])
                        )

    def test_all_instances_realizable(self) -> None:
        for scenario in self.config["scenarios"]:
            with self.subTest(name=scenario["name"]):
                self.assertTrue(is_realizable(scenario))

    def test_methods_are_all_non_linear(self) -> None:
        for name, spec in self.config["methods"].items():
            with self.subTest(method=name):
                self.assertIn(
                    spec["algorithm"],
                    {"mlp_projected_group_omd", "mlp_rwp_omd"},
                )


class AlphaEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load("stage_b_mlp.json")

    def test_calibration_gate_passes_on_the_linear_head(self) -> None:
        result = run_calibration(self.config)
        self.assertEqual(result["outcome"], "PASS")
        self.assertLess(result["max_abs_error"], 1e-9)

    def test_alpha_is_one_for_complete_aliasing_under_the_linear_head(self) -> None:
        name = f"{self.config['control_base']}_eps000"
        scenario = next(
            s for s in self.config["scenarios"] if s["name"] == name
        )
        method = dict(self.config["methods"]["uniform_eta075000"])
        method["algorithm"] = "projected_group_omd"
        method.pop("mlp_hidden", None)
        algorithm = build_algorithm("c", method, scenario, 0)
        critical = set(scenario["critical_positions"])
        signs = np.array([
            1.0 if k in critical else -1.0 for k in range(scenario["horizon"])
        ])
        self.assertAlmostEqual(
            alpha_from_jacobians(position_jacobians(algorithm), signs), 1.0, places=12
        )

    def test_alpha_is_bounded_and_finite_under_the_mlp(self) -> None:
        scenario = self.config["scenarios"][0]
        method = self.config["methods"]["uniform_eta075000"]
        result = measure_alpha(scenario, method, 700, 100)
        self.assertTrue(np.isfinite(result["alpha_mean"]))
        self.assertGreaterEqual(result["alpha_mean"], -1.0)
        self.assertLessEqual(result["alpha_mean"], 1.0)
        self.assertEqual(result["n_undefined"], 0)

    def test_alpha_uses_only_the_baseline_arm(self) -> None:
        """Non-circularity: the candidate must not appear in alpha's inputs."""

        rule = self.config["decision_rule"]
        baseline = self.config["methods"][rule["baseline_alpha_method"]]
        self.assertEqual(baseline["algorithm"], "mlp_projected_group_omd")
        self.assertNotEqual(
            rule["baseline_alpha_method"], rule["candidate_method"]
        )


class ConfirmationRuleTest(unittest.TestCase):
    """The direction-agnostic criterion, per E11."""

    def _config(self, sign: int):
        names = [f"c{i}" for i in range(12)]
        return {
            "protocol_id": "stage-b-mlp-2026-08-19",
            "eps_levels": [0.0],
            "split": {"discovery": [], "confirmation": names},
            "scenarios": [
                {"name": n, "feature_noise": 0.0, "seeds": [0],
                 "critical_positions": [0], "horizon": 2}
                for n in names
            ],
            "methods": {},
            "training": {"evaluation_interval": 5},
            "decision_rule": {
                "candidate_method": "rwp", "baseline_alpha_method": "uni",
                "grid_prefix": "uniform_eta", "max_dropped_seeds": 2,
                "expected_sign": sign, "confirmation_magnitude_bound": 0.50,
                "bootstrap_resamples": 300, "bootstrap_seed": 20260822,
            },
        }

    def _patched(self, config, alphas, deltas, monkey):
        names = config["split"]["confirmation"]
        monkey["metrics"] = {
            n: {"usable": True, "delta_corrected": d}
            for n, d in zip(names, deltas)
        }
        monkey["alpha"] = {
            n: {"alpha": a} for n, a in zip(names, alphas)
        }

    def test_strongly_negative_correlation_confirms(self) -> None:
        import experiments.evaluate_stage_b as module

        config = self._config(-1)
        names = config["split"]["confirmation"]
        alphas = np.linspace(0.0, 1.0, len(names))
        deltas = -alphas * 0.05 + 0.06
        original = module.measure
        module.measure = lambda c, s, n: (
            {k: {"usable": True, "delta_corrected": float(d)}
             for k, d in zip(names, deltas)},
            {k: {"alpha": float(a)} for k, a in zip(names, alphas)},
        )
        try:
            result = run_confirmation(config, {})
        finally:
            module.measure = original
        self.assertLess(result["spearman"]["rho"], 0.0)
        self.assertEqual(result["outcome"], "CONFIRMED")
        self.assertIn("reinstated", result["consequence"])

    def test_wrong_direction_does_not_confirm(self) -> None:
        import experiments.evaluate_stage_b as module

        config = self._config(-1)
        names = config["split"]["confirmation"]
        alphas = np.linspace(0.0, 1.0, len(names))
        deltas = alphas * 0.05  # positive relationship, opposite of expected
        original = module.measure
        module.measure = lambda c, s, n: (
            {k: {"usable": True, "delta_corrected": float(d)}
             for k, d in zip(names, deltas)},
            {k: {"alpha": float(a)} for k, a in zip(names, alphas)},
        )
        try:
            result = run_confirmation(config, {})
        finally:
            module.measure = original
        self.assertFalse(result["sign_matches_expected"])
        self.assertEqual(result["outcome"], "NOT_CONFIRMED")
        self.assertIn("PROHIBITED", result["consequence"])


if __name__ == "__main__":
    unittest.main()
