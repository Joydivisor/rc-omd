"""Mechanical checks for the frozen shared-parameter protocol."""

from __future__ import annotations

import unittest

from experiments.evaluate_function_approx_protocol import evaluate


class FunctionApproxProtocolTest(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "protocol_id": "function-approx-v1-2026-08-09",
            "scenarios": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "decision_rule": {
                "primary_uniform": "u",
                "primary_online": "o",
                "max_auc_deficit": 0.02,
                "max_distractor_kl_ratio": 0.75,
                "max_runtime_ratio": 1.5,
                "required_scenario_passes": 2,
                "required_runtime_passes": 3,
            },
        }

    @staticmethod
    def _method(auc: float, kl: float, runtime: float) -> dict:
        return {
            "success_auc_mean": auc,
            "cumulative_distractor_kl_mean": kl,
            "runtime_seconds_mean": runtime,
        }

    def test_transfer_rule_is_applied_mechanically(self) -> None:
        summary = {
            "a": {"u": self._method(0.9, 1.0, 1.0), "o": self._method(0.89, 0.7, 1.2)},
            "b": {"u": self._method(0.9, 1.0, 1.0), "o": self._method(0.88, 0.75, 1.4)},
            "c": {"u": self._method(0.9, 1.0, 1.0), "o": self._method(0.86, 0.5, 1.1)},
        }
        result = evaluate(self._config(), summary)
        self.assertEqual(result["scenario_pass_count"], 2)
        self.assertEqual(result["transfer_decision"], "GO")
        self.assertEqual(result["systems_feasibility_decision"], "PASS")

    def test_scenario_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(self._config(), {"a": {}, "b": {}})


if __name__ == "__main__":
    unittest.main()
