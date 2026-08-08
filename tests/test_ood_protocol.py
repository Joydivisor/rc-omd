from __future__ import annotations

import unittest

from experiments.evaluate_ood_protocol import PROTOCOL_ID, evaluate


class OODProtocolEvaluationTest(unittest.TestCase):
    def test_go_rule_is_applied_mechanically(self) -> None:
        names = ["a", "b", "c", "d"]
        config = {
            "protocol_id": PROTOCOL_ID,
            "scenarios": [{"name": name} for name in names],
        }
        summary = {
            name: {
                "uniform_eta075": {
                    "success_auc_mean": 0.95,
                    "cumulative_distractor_kl_mean": 0.1,
                    "runtime_seconds_mean": 1.0,
                },
                "online_eta125": {
                    "success_auc_mean": 0.945,
                    "cumulative_distractor_kl_mean": 0.04,
                    "runtime_seconds_mean": 1.1,
                },
            }
            for name in names
        }

        result = evaluate(config, summary)

        self.assertEqual(result["scenario_pass_count"], 4)
        self.assertEqual(result["generalization_decision"], "GO")
        self.assertEqual(result["systems_feasibility_decision"], "PASS")

    def test_scenario_mismatch_is_rejected(self) -> None:
        config = {"protocol_id": PROTOCOL_ID, "scenarios": [{"name": "a"}]}
        with self.assertRaises(ValueError):
            evaluate(config, {})


if __name__ == "__main__":
    unittest.main()
