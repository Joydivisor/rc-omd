from __future__ import annotations

import unittest

import numpy as np

from experiments.evaluate_pareto_protocol import evaluate, t_ci_95


SEEDS = list(range(20))
RULE = {
    "step_grid": [0.50, 1.00],
    "uniform_method_prefix": "uniform_eta",
    "online_method_prefix": "online_eta",
    "auc_non_inferiority_margin": 0.01,
    "distractor_kl_ratio_bound": 0.75,
    "degenerate_kl_threshold": 1e-9,
    "max_dropped_kl_seeds": 2,
    "required_covered_points_per_scenario": 2,
    "total_points_per_scenario": 2,
    "required_scenario_passes": 1,
    "total_scenarios": 1,
    "power_failure_ci_half_width_fraction": 0.25,
    "max_runtime_ratio": 1.5,
}
CONFIG = {
    "protocol_id": "pareto-v1-2026-08-14",
    "scenarios": [{"name": "toy"}],
    "decision_rule": RULE,
}


def method_entry(auc: float, kl: float, runtime: float = 1.0) -> dict[str, object]:
    return {
        "seeds": SEEDS,
        "success_auc_per_seed": [auc] * len(SEEDS),
        "cumulative_distractor_kl_per_seed": [kl] * len(SEEDS),
        "runtime_seconds_per_seed": [runtime] * len(SEEDS),
    }


class TCriticalTest(unittest.TestCase):
    def test_zero_variance_gives_zero_half_width(self) -> None:
        ci = t_ci_95(np.full(20, 0.01))
        self.assertAlmostEqual(ci["mean"], 0.01)
        self.assertAlmostEqual(ci["half_width"], 0.0)

    def test_rejects_untabulated_df(self) -> None:
        with self.assertRaises(ValueError):
            t_ci_95(np.zeros(5))


class EvaluateParetoProtocolTest(unittest.TestCase):
    def test_go_when_every_point_is_covered(self) -> None:
        summary = {
            "toy": {
                "uniform_eta050": method_entry(auc=0.80, kl=1.0),
                "uniform_eta100": method_entry(auc=0.78, kl=1.2),
                "online_eta050": method_entry(auc=0.81, kl=0.5),
                "online_eta100": method_entry(auc=0.83, kl=0.6),
            }
        }
        result = evaluate(CONFIG, summary)
        report = result["scenario_reports"]["toy"]
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["covered_count"], 2)
        self.assertEqual(result["decision"], "GO")
        self.assertEqual(result["systems_feasibility_decision"], "PASS")
        for point in report["points"]:
            self.assertEqual(point["matched_online_step"], 0.50)

    def test_no_go_when_no_online_step_matches(self) -> None:
        summary = {
            "toy": {
                "uniform_eta050": method_entry(auc=0.90, kl=1.0),
                "uniform_eta100": method_entry(auc=0.90, kl=1.0),
                "online_eta050": method_entry(auc=0.50, kl=0.5),
                "online_eta100": method_entry(auc=0.50, kl=0.5),
            }
        }
        result = evaluate(CONFIG, summary)
        report = result["scenario_reports"]["toy"]
        self.assertEqual(report["covered_count"], 0)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(result["decision"], "NO-GO")
        for point in report["points"]:
            self.assertIsNone(point["matched_online_step"])
            self.assertFalse(point["covered"])

    def test_no_go_when_kl_condition_fails_at_matched_step(self) -> None:
        summary = {
            "toy": {
                "uniform_eta050": method_entry(auc=0.80, kl=1.0),
                "uniform_eta100": method_entry(auc=0.80, kl=1.0),
                # AUC passes immediately at the smallest step, but KL ratio
                # 0.9 > bound 0.75, so the point must be uncovered even
                # though a step was matched.
                "online_eta050": method_entry(auc=0.85, kl=0.9),
                "online_eta100": method_entry(auc=0.85, kl=0.9),
            }
        }
        result = evaluate(CONFIG, summary)
        report = result["scenario_reports"]["toy"]
        self.assertEqual(report["covered_count"], 0)
        for point in report["points"]:
            self.assertEqual(point["matched_online_step"], 0.50)
            self.assertFalse(point["covered"])
            self.assertFalse(point["kl"]["kl_pass"])

    def test_matching_rule_prefers_smallest_qualifying_step(self) -> None:
        summary = {
            "toy": {
                "uniform_eta050": method_entry(auc=0.80, kl=1.0),
                "uniform_eta100": method_entry(auc=0.80, kl=1.0),
                # Fails at 0.50, passes at 1.00.
                "online_eta050": method_entry(auc=0.70, kl=0.5),
                "online_eta100": method_entry(auc=0.82, kl=0.5),
            }
        }
        result = evaluate(CONFIG, summary)
        report = result["scenario_reports"]["toy"]
        for point in report["points"]:
            self.assertEqual(point["matched_online_step"], 1.00)
            self.assertTrue(point["frontier_truncated"])

    def test_kl_test_drops_degenerate_seeds_up_to_the_limit(self) -> None:
        uniform_kl = [1.0] * 18 + [0.0, 0.0]
        online_kl = [0.5] * 20
        summary = {
            "toy": {
                "uniform_eta050": {
                    "seeds": SEEDS,
                    "success_auc_per_seed": [0.80] * 20,
                    "cumulative_distractor_kl_per_seed": uniform_kl,
                    "runtime_seconds_per_seed": [1.0] * 20,
                },
                "uniform_eta100": method_entry(auc=0.80, kl=1.0),
                "online_eta050": {
                    "seeds": SEEDS,
                    "success_auc_per_seed": [0.85] * 20,
                    "cumulative_distractor_kl_per_seed": online_kl,
                    "runtime_seconds_per_seed": [1.0] * 20,
                },
                "online_eta100": method_entry(auc=0.85, kl=0.5),
            }
        }
        result = evaluate(CONFIG, summary)
        point = result["scenario_reports"]["toy"]["points"][0]
        self.assertEqual(point["kl"]["dropped_seeds"], 2)
        self.assertFalse(point["kl"]["inconclusive"])
        self.assertTrue(point["covered"])

    def test_kl_test_is_inconclusive_past_the_drop_limit(self) -> None:
        uniform_kl = [1.0] * 17 + [0.0, 0.0, 0.0]
        summary = {
            "toy": {
                "uniform_eta050": {
                    "seeds": SEEDS,
                    "success_auc_per_seed": [0.80] * 20,
                    "cumulative_distractor_kl_per_seed": uniform_kl,
                    "runtime_seconds_per_seed": [1.0] * 20,
                },
                "uniform_eta100": method_entry(auc=0.80, kl=1.0),
                "online_eta050": method_entry(auc=0.85, kl=0.5),
                "online_eta100": method_entry(auc=0.85, kl=0.5),
            }
        }
        result = evaluate(CONFIG, summary)
        point = result["scenario_reports"]["toy"]["points"][0]
        self.assertTrue(point["kl"]["inconclusive"])
        self.assertFalse(point["covered"])

    def test_seed_mismatch_between_paired_methods_is_rejected(self) -> None:
        summary = {
            "toy": {
                "uniform_eta050": method_entry(auc=0.80, kl=1.0),
                "uniform_eta100": method_entry(auc=0.80, kl=1.0),
                "online_eta050": {**method_entry(auc=0.85, kl=0.5), "seeds": SEEDS[::-1]},
                "online_eta100": method_entry(auc=0.85, kl=0.5),
            }
        }
        with self.assertRaises(ValueError):
            evaluate(CONFIG, summary)


if __name__ == "__main__":
    unittest.main()
