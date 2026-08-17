from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_geometry_protocol import (
    check_invariant,
    evaluate_scenario,
    run_decision,
    run_selection,
)


SEEDS = list(range(20))
RULE = {
    "auc_non_inferiority_margin": 0.02,
    "distractor_kl_ratio_bound": 0.75,
    "degenerate_kl_threshold": 1e-9,
    "max_dropped_kl_seeds": 2,
    "auc_precision_floor": 0.02,
    "log_kl_precision_floor": 0.6931471805599453,
    "invariant_tolerance": 1e-9,
}


def entry(auc, kl, critical=None, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = len(SEEDS)
    a = np.full(n, auc) + (rng.normal(size=n) * jitter if jitter else 0.0)
    k = np.full(n, kl)
    return {
        "seeds": SEEDS,
        "success_auc_per_seed": a.tolist(),
        "cumulative_distractor_kl_per_seed": k.tolist(),
        "cumulative_critical_kl_per_seed": (
            [critical] * n if critical is not None else k.tolist()
        ),
        "runtime_seconds_per_seed": [1.0] * n,
    }


class ScenarioVerdictTest(unittest.TestCase):
    def test_pass_when_both_conditions_hold(self) -> None:
        r = evaluate_scenario("s", entry(0.80, 1.0), entry(0.81, 0.5), RULE)
        self.assertEqual(r["status"], "pass")

    def test_fail_when_kl_bound_missed_with_tight_intervals(self) -> None:
        r = evaluate_scenario("s", entry(0.80, 1.0), entry(0.81, 0.9), RULE)
        self.assertEqual(r["status"], "fail")
        self.assertFalse(r["kl"]["kl_pass"])

    def test_fail_when_auc_drops_below_margin(self) -> None:
        r = evaluate_scenario("s", entry(0.80, 1.0), entry(0.70, 0.1), RULE)
        self.assertEqual(r["status"], "fail")
        self.assertFalse(r["auc"]["auc_pass"])

    def test_wide_interval_never_converts_a_pass_into_inconclusive(self) -> None:
        """The power clause must apply only to failures: both conditions are
        conservative under widening, so a wide CI cannot manufacture a pass.
        Independent jitter seeds are required, otherwise the noise cancels in
        the paired difference and the interval is not actually wide."""
        base = entry(0.80, 1.0, jitter=0.08, seed=1)
        cand = entry(0.95, 0.2, jitter=0.08, seed=2)
        r = evaluate_scenario("s", base, cand, RULE)
        self.assertGreater(
            r["auc"]["d_auc"]["half_width"], 0.02, "interval is not wide enough to test"
        )
        self.assertEqual(r["status"], "pass")

    def test_imprecise_auc_failure_is_inconclusive_not_fail(self) -> None:
        base = entry(0.80, 1.0, jitter=0.40, seed=2)
        cand = entry(0.60, 0.1, jitter=0.40, seed=3)
        r = evaluate_scenario("s", base, cand, RULE)
        self.assertGreater(r["auc"]["d_auc"]["half_width"], 0.02)
        self.assertEqual(r["status"], "inconclusive")

    def test_excessive_degenerate_seeds_is_inconclusive(self) -> None:
        base = entry(0.80, 1.0)
        base["cumulative_distractor_kl_per_seed"] = [0.0] * 5 + [1.0] * 15
        r = evaluate_scenario("s", base, entry(0.81, 0.5), RULE)
        self.assertTrue(r["kl"]["degenerate"])
        self.assertEqual(r["status"], "inconclusive")

    def test_droppable_degenerate_seeds_still_adjudicate(self) -> None:
        base = entry(0.80, 1.0)
        base["cumulative_distractor_kl_per_seed"] = [0.0] * 2 + [1.0] * 18
        r = evaluate_scenario("s", base, entry(0.81, 0.5), RULE)
        self.assertEqual(r["kl"]["dropped_seeds"], 2)
        self.assertFalse(r["kl"]["degenerate"])
        self.assertEqual(r["status"], "pass")

    def test_seed_mismatch_is_rejected(self) -> None:
        cand = entry(0.81, 0.5)
        cand["seeds"] = SEEDS[::-1]
        with self.assertRaises(ValueError):
            evaluate_scenario("s", entry(0.80, 1.0), cand, RULE)


class InvariantTest(unittest.TestCase):
    def test_equal_critical_and_distractor_kl_holds(self) -> None:
        methods = {"m": entry(0.8, 0.5, critical=0.5)}
        out = check_invariant("complete", methods, ["m"], 1e-9)
        self.assertTrue(out["invariant_holds"])

    def test_unequal_kl_violates(self) -> None:
        methods = {"m": entry(0.8, 0.5, critical=0.9)}
        out = check_invariant("complete", methods, ["m"], 1e-9)
        self.assertFalse(out["invariant_holds"])


def dev_config():
    grid = [
        {"name": "rwp_lam100_mu0", "lambda": 1.0, "mu": 0.0},
        {"name": "rwp_lam200_mu0", "lambda": 2.0, "mu": 0.0},
        {"name": "rwp_lam300_mu0", "lambda": 3.0, "mu": 0.0},
    ]
    return {
        "protocol_id": "geometry-v1-2026-08-15",
        "selection": {
            **RULE,
            "baseline_method": "base",
            "grid": grid,
            "negative_check_method": "neg",
            "regression_scenario": "sep",
            "selection_scenarios": ["sep", "partial"],
            "invariant_scenarios": ["complete"],
        },
    }


def dev_summary(lam200_partial_kl=0.5, lam100_sep_auc=0.81, complete_critical=0.5):
    base = lambda: entry(0.80, 1.0, critical=1.0)
    return {
        "sep": {
            "base": base(),
            "rwp_lam100_mu0": entry(lam100_sep_auc, 0.5),
            "rwp_lam200_mu0": entry(0.81, 0.5),
            "rwp_lam300_mu0": entry(0.81, 0.4),
            "neg": entry(0.81, 0.5),
        },
        "partial": {
            "base": base(),
            "rwp_lam100_mu0": entry(0.81, 0.95),
            "rwp_lam200_mu0": entry(0.81, lam200_partial_kl),
            "rwp_lam300_mu0": entry(0.81, 0.5),
            "neg": entry(0.81, 0.95),
        },
        "complete": {
            "base": base(),
            "rwp_lam100_mu0": entry(0.81, 0.5, critical=complete_critical),
            "rwp_lam200_mu0": entry(0.81, 0.5, critical=complete_critical),
            "rwp_lam300_mu0": entry(0.81, 0.5, critical=complete_critical),
            "neg": entry(0.81, 0.5, critical=complete_critical),
        },
    }


class SelectionRuleTest(unittest.TestCase):
    def test_selects_highest_pass_count(self) -> None:
        out = run_selection(dev_config(), dev_summary())
        self.assertEqual(out["outcome"], "SELECTED")
        # lam100 fails partial (0.95 > 0.75); lam200 and lam300 both pass both.
        self.assertEqual(out["selected"]["pass_count"], 2)

    def test_tie_broken_towards_lower_kl_then_smaller_lambda(self) -> None:
        """lam200 and lam300 both pass twice; lam300 has the lower mean KL
        bound (0.4/0.5 vs 0.5/0.5) so it must win on tie-break 1, ahead of the
        smaller-lambda rule."""
        out = run_selection(dev_config(), dev_summary())
        self.assertEqual(out["selected"]["method"], "rwp_lam300_mu0")

    def test_smaller_lambda_wins_when_kl_bounds_are_equal(self) -> None:
        summary = dev_summary()
        summary["sep"]["rwp_lam300_mu0"] = entry(0.81, 0.5)  # equalise
        out = run_selection(dev_config(), summary)
        self.assertEqual(out["selected"]["method"], "rwp_lam200_mu0")

    def test_regression_gate_excludes_candidates(self) -> None:
        summary = dev_summary(lam100_sep_auc=0.50)  # lam100 regresses separable
        out = run_selection(dev_config(), summary)
        self.assertFalse(
            next(c for c in out["candidates"] if c["name"] == "rwp_lam100_mu0")[
                "passes_regression_gate"
            ]
        )
        self.assertEqual(out["eligible_count"], 2)

    def test_no_eligible_candidate_terminates_without_a_test_run(self) -> None:
        summary = dev_summary()
        for name in ("rwp_lam100_mu0", "rwp_lam200_mu0", "rwp_lam300_mu0"):
            summary["sep"][name] = entry(0.50, 0.5)
        out = run_selection(dev_config(), summary)
        self.assertIsNone(out["selected"])
        self.assertEqual(out["outcome"], "NO-GO_NO_ELIGIBLE_CANDIDATE")

    def test_invariant_violation_halts_selection(self) -> None:
        out = run_selection(dev_config(), dev_summary(complete_critical=0.9))
        self.assertFalse(out["invariant_holds"])
        self.assertEqual(out["outcome"], "HALT_INVARIANT_VIOLATED")

    def test_negative_check_is_never_selected(self) -> None:
        summary = dev_summary()
        summary["partial"]["neg"] = entry(0.99, 0.01)  # make it look best
        summary["sep"]["neg"] = entry(0.99, 0.01)
        out = run_selection(dev_config(), summary)
        self.assertNotEqual(out["selected"]["method"], "neg")

    def test_negative_check_beating_the_grid_is_flagged(self) -> None:
        """lambda < 1 weights unreliable positions less, so it should not win.
        If it does, the implementation or the objective is suspect and the flag
        must surface that rather than letting it pass unnoticed."""
        summary = dev_summary()
        for name in ("rwp_lam100_mu0", "rwp_lam200_mu0", "rwp_lam300_mu0"):
            summary["partial"][name] = entry(0.81, 0.95)  # every grid point fails
        summary["partial"]["neg"] = entry(0.81, 0.01)
        out = run_selection(dev_config(), summary)
        self.assertEqual(out["negative_check"]["pass_count"], 2)
        self.assertTrue(out["negative_check"]["outperformed_grid"])
        self.assertNotEqual(out["selected"]["method"], "neg")


def test_config():
    return {
        "protocol_id": "geometry-v1-2026-08-15",
        "decision_rule": {
            **RULE,
            "baseline_method": "base",
            "candidate_method": "cand",
            "required_scenarios": ["separable", "partial", "holdout_a", "holdout_b"],
            "invariant_scenarios": ["complete"],
        },
    }


def test_summary(**overrides):
    scenarios = {
        name: {"base": entry(0.80, 1.0, critical=1.0), "cand": entry(0.81, 0.5)}
        for name in ("separable", "partial", "holdout_a", "holdout_b")
    }
    scenarios["complete"] = {
        "base": entry(0.80, 1.0, critical=1.0),
        "cand": entry(0.81, 0.5, critical=0.5),
    }
    for key, value in overrides.items():
        scenarios[key]["cand"] = value
    return scenarios


class DecisionRuleTest(unittest.TestCase):
    def test_go_requires_all_four(self) -> None:
        out = run_decision(test_config(), test_summary())
        self.assertEqual(out["decision"], "GO")
        self.assertEqual(out["pass_count"], 4)

    def test_single_failure_is_no_go(self) -> None:
        out = run_decision(test_config(), test_summary(holdout_b=entry(0.81, 0.95)))
        self.assertEqual(out["decision"], "NO-GO")
        self.assertEqual(out["pass_count"], 3)

    def test_separable_regression_alone_is_no_go(self) -> None:
        out = run_decision(test_config(), test_summary(separable=entry(0.81, 0.95)))
        self.assertEqual(out["decision"], "NO-GO")

    def test_inconclusive_scenario_makes_the_whole_result_inconclusive(self) -> None:
        noisy = entry(0.60, 0.1, jitter=0.40, seed=11)
        out = run_decision(test_config(), test_summary(partial=noisy))
        self.assertEqual(out["decision"], "INCONCLUSIVE")

    def test_invariant_violation_halts_before_any_decision(self) -> None:
        summary = test_summary()
        summary["complete"]["cand"] = entry(0.81, 0.5, critical=0.9)
        out = run_decision(test_config(), summary)
        self.assertEqual(out["decision"], "HALT_INVARIANT_VIOLATED")


class CommittedDevConfigTest(unittest.TestCase):
    """The committed config must match the protocol's declared tie-groups."""

    DECLARED_ALPHA = {
        "geom_dev_separable": 0.0,
        "geom_dev_partial_040": 0.4,
        "geom_dev_partial_080": 0.8,
        "geom_dev_complete": 1.0,
    }

    def setUp(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "geometry_dev.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def test_alpha_matches_the_protocol_table(self) -> None:
        for scenario in self.config["scenarios"]:
            F = np.asarray(scenario["features"], dtype=float)
            H = int(scenario["horizon"])
            crit = set(scenario["critical_positions"])
            groups: dict[tuple, list[int]] = {}
            for k in range(H):
                groups.setdefault(tuple(F[k]), []).append(k)
            spread = sum(
                abs(sum(1 for m in g if m in crit) - sum(1 for m in g if m not in crit))
                for g in groups.values()
            )
            alpha = 1 - spread / H
            self.assertAlmostEqual(
                alpha, self.DECLARED_ALPHA[scenario["name"]], places=9,
                msg=f"{scenario['name']} alpha drifted from the protocol table",
            )

    def test_scenarios_are_realizable(self) -> None:
        """Tied critical positions must share a target action, or the optimum
        is unrepresentable and the scenario cannot reach its threshold."""
        for scenario in self.config["scenarios"]:
            F = np.asarray(scenario["features"], dtype=float)
            H = int(scenario["horizon"])
            targets = dict(
                zip(scenario["critical_positions"], scenario["target_actions"])
            )
            groups: dict[tuple, list[int]] = {}
            for k in range(H):
                groups.setdefault(tuple(F[k]), []).append(k)
            achievable = 0
            for g in groups.values():
                ts = [targets[m] for m in g if m in targets]
                if ts:
                    achievable += max(ts.count(t) for t in set(ts))
            self.assertGreaterEqual(
                achievable, int(scenario["minimum_matches"]),
                msg=f"{scenario['name']} cannot reach its threshold",
            )

    def test_grid_matches_the_frozen_specification(self) -> None:
        grid = self.config["selection"]["grid"]
        self.assertEqual(len(grid), 15)
        self.assertEqual(
            sorted({p["lambda"] for p in grid}), [1.0, 1.5, 2.0, 3.0, 5.0]
        )
        self.assertEqual(sorted({p["mu"] for p in grid}), [0.0, 1e-3, 1e-2])
        self.assertTrue(all(p["lambda"] >= 1.0 for p in grid))
        self.assertTrue(any(p["mu"] > 0.0 for p in grid))

    def test_every_referenced_method_exists(self) -> None:
        methods = self.config["methods"]
        selection = self.config["selection"]
        for name in [p["name"] for p in selection["grid"]] + [
            selection["baseline_method"],
            selection["v1_reference_method"],
            selection["negative_check_method"],
        ]:
            self.assertIn(name, methods)


if __name__ == "__main__":
    unittest.main()
