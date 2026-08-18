"""Structural checks for the geometry-v2 scenario set.

These assert properties that `docs/GEOMETRY_V2_PROTOCOL.md` treats as
requirements on the experiment rather than as ordinary code behaviour: if an
A3 arm stops being matched, or an arm stops being realizable, the experiment
no longer tests what it claims to test.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiments.geometry_v2_scenarios import (
    A3_PROFILES,
    MATCHED_PROPERTIES,
    alpha_is_admissible,
    build_a3_triple,
    build_scenario,
    is_realizable,
    structure,
    tie_groups,
    validate_a3_triple,
)

REPO = Path(__file__).resolve().parent.parent


class A3TripleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.triple = build_a3_triple()
        self.by_name = {s["name"]: s for s in self.triple}

    def test_arms_agree_on_every_matched_property(self) -> None:
        report = validate_a3_triple(self.triple)
        for prop in MATCHED_PROPERTIES:
            with self.subTest(prop=prop):
                self.assertTrue(
                    report["matched_properties"][prop],
                    f"A3 arms disagree on {prop}",
                )
        self.assertTrue(report["all_matched"])

    def test_homog_and_pure_arms_are_exactly_matched(self) -> None:
        """The clean pure_crit test: identical everywhere except pure_crit.

        These two additionally agree on the per-group critical-count multiset,
        which the heterogeneous arm structurally cannot.
        """

        report = validate_a3_triple(self.triple)
        self.assertTrue(report["exact_pair_matched"])
        homog = structure(self.by_name["geom_v2_a3_pure0_homog"])
        pure = structure(self.by_name["geom_v2_a3_pure_ge1"])
        self.assertEqual(homog["critical_counts"], pure["critical_counts"])
        self.assertNotEqual(homog["pure_crit"], pure["pure_crit"])

    def test_pure_crit_and_ratio_class(self) -> None:
        expected = {
            "geom_v2_a3_pure0_homog": (0, "homogeneous"),
            "geom_v2_a3_pure0_hetero": (0, "heterogeneous"),
            "geom_v2_a3_pure_ge1": (1, "heterogeneous"),
        }
        for name, (pure_crit, ratio_class) in expected.items():
            with self.subTest(name=name):
                info = structure(self.by_name[name])
                self.assertEqual(info["pure_crit"], pure_crit)
                self.assertEqual(info["ratio_class"], ratio_class)

    def test_alpha_is_one_third_in_every_arm(self) -> None:
        for scenario in self.triple:
            with self.subTest(name=scenario["name"]):
                self.assertAlmostEqual(structure(scenario)["alpha"], 1 / 3, places=12)

    def test_heterogeneous_arm_cannot_match_critical_counts(self) -> None:
        """Documents why the discriminator is the weaker arm.

        Heterogeneous finite yield ratios at a fixed group-size multiset force
        unequal critical counts, so this mismatch is structural rather than an
        oversight in the scenario design.
        """

        hetero = structure(self.by_name["geom_v2_a3_pure0_hetero"])
        homog = structure(self.by_name["geom_v2_a3_pure0_homog"])
        self.assertNotEqual(hetero["critical_counts"], homog["critical_counts"])
        self.assertEqual(hetero["group_sizes"], homog["group_sizes"])

    def test_all_arms_realizable(self) -> None:
        for scenario in self.triple:
            with self.subTest(name=scenario["name"]):
                self.assertTrue(is_realizable(scenario))

    def test_features_partition_all_positions_one_hot(self) -> None:
        for scenario in self.triple:
            with self.subTest(name=scenario["name"]):
                horizon = scenario["horizon"]
                covered = sorted(k for g in tie_groups(scenario) for k in g)
                self.assertEqual(covered, list(range(horizon)))
                for row in scenario["features"]:
                    self.assertEqual(sum(row), 1)

    def test_generation_is_deterministic(self) -> None:
        again = build_a3_triple()
        self.assertEqual(json.dumps(self.triple), json.dumps(again))

    def test_critical_positions_precede_distractors_within_group(self) -> None:
        for name, profile in A3_PROFILES.items():
            scenario = build_scenario(name, profile)
            critical = set(scenario["critical_positions"])
            for members in tie_groups(scenario):
                flags = [k in critical for k in sorted(members)]
                self.assertEqual(flags, sorted(flags, reverse=True))

    def test_profile_must_cover_the_frozen_horizon(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario("bad", ((1, 1),))


class ParityLemmaTest(unittest.TestCase):
    """Lemma 1: alpha is quantized by 2/H, so alpha=1/2 needs 4 | H."""

    def test_alpha_half_unreachable_at_h10_reachable_at_h12(self) -> None:
        self.assertFalse(alpha_is_admissible(10, 0.5))
        self.assertTrue(alpha_is_admissible(12, 0.5))
        self.assertTrue(alpha_is_admissible(16, 0.5))

    def test_a3_alpha_is_admissible(self) -> None:
        self.assertTrue(alpha_is_admissible(12, 1 / 3))

    def test_non_quantized_alpha_rejected(self) -> None:
        self.assertFalse(alpha_is_admissible(12, 0.4))


class InheritedScenarioTest(unittest.TestCase):
    """The five inherited scenarios must not drift from geometry-v1."""

    def setUp(self) -> None:
        self.v1 = json.loads(
            (REPO / "configs" / "geometry_v1.json").read_text(encoding="utf-8")
        )
        self.v2 = json.loads(
            (REPO / "configs" / "geometry_v2.json").read_text(encoding="utf-8")
        )

    def test_frozen_v1_config_matches_golden_record_hash(self) -> None:
        record = json.loads(
            (REPO / "paper" / "frozen" / "geometry-v1-2026-08-15.json").read_text(
                encoding="utf-8"
            )
        )
        payload = (REPO / record["config"]["path"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), record["config"]["sha256"]
        )

    def test_inherited_scenarios_copied_verbatim(self) -> None:
        v1_by_name = {s["name"]: s for s in self.v1["scenarios"]}
        inherited = [
            "separable_shared_features",
            "geom_holdout_a",
            "geom_holdout_b",
            "partial_feature_aliasing",
            "complete_feature_aliasing_negative_control",
        ]
        v2_by_name = {s["name"]: s for s in self.v2["scenarios"]}
        for name in inherited:
            with self.subTest(name=name):
                self.assertEqual(
                    json.dumps(v1_by_name[name], sort_keys=True),
                    json.dumps(v2_by_name[name], sort_keys=True),
                )

    def test_recorded_structure_matches_protocol_table(self) -> None:
        expected = {
            "separable_shared_features": (3, "homogeneous"),
            "geom_holdout_a": (1, "heterogeneous"),
            "geom_holdout_b": (1, "heterogeneous"),
            "partial_feature_aliasing": (0, "homogeneous"),
            "complete_feature_aliasing_negative_control": (0, "homogeneous"),
        }
        for name, (pure_crit, ratio_class) in expected.items():
            with self.subTest(name=name):
                info = self.v2["scenario_structure"][name]
                self.assertEqual(info["pure_crit"], pure_crit)
                self.assertEqual(info["ratio_class"], ratio_class)

    def test_every_pure_crit_zero_scenario_measured_so_far_is_homogeneous(self) -> None:
        """The confound this protocol exists to break.

        If this ever fails, an untested cell has appeared in the inherited set
        and the discriminator arm is no longer the only one.
        """

        for name, info in self.v2["scenario_structure"].items():
            if name.startswith("geom_v2_a3"):
                continue
            if info["pure_crit"] == 0:
                self.assertEqual(info["ratio_class"], "homogeneous", name)


if __name__ == "__main__":
    unittest.main()
