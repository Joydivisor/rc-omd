"""Pairing assertions required by docs/PARETO_V1_PROTOCOL.md.

The protocol's paired-difference statistics (d_AUC(s), r_KL(s)) are only valid
if seed `s` produces identical environment stochasticity for every method and
every step size. This is asserted here rather than assumed, per the protocol's
"Frozen design > Pairing" clause.
"""

from __future__ import annotations

import unittest

import numpy as np

from experiments.run_reliability_diagnostics import (
    build_algorithm,
    build_environment,
    run_one,
    summarize,
)


SCENARIO = {
    "name": "pairing_probe",
    "environment": "controlled_all",
    "horizon": 4,
    "n_actions": 2,
    "critical_positions": [1, 3],
    "target_actions": [1, 0],
    "initial_policy": [[0.5, 0.5]] * 4,
    "iterations": 6,
    "group_size": 8,
}

STEP_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)

METHOD_CONFIGS: dict[str, dict[str, object]] = {}
for step in STEP_GRID:
    key = f"uniform_eta{int(round(step * 100)):03d}"
    METHOD_CONFIGS[key] = {"algorithm": "uniform_group_omd", "step_size": step}
for step in STEP_GRID:
    key = f"online_eta{int(round(step * 100)):03d}"
    METHOD_CONFIGS[key] = {
        "algorithm": "online_rc_omd",
        "step_size": step,
        "reliability_decay": 0.9,
        "confidence_multiplier": 1.0,
        "warmup_effective_samples": 8.0,
        "reliability_floor": 0.1,
    }


class ParetoPairingTest(unittest.TestCase):
    def test_first_batch_identical_across_methods_and_step_sizes_at_fixed_seed(
        self,
    ) -> None:
        """At iteration 0 every method shares the scenario's `initial_policy`,
        so this is the cleanest point to check that `run_one`'s environment RNG
        depends on `seed` alone. If any method or step-size component leaked
        into the seed (the class of defect E5 fixed for Milestone 2), the first
        sampled batch would diverge here."""
        seed = 7
        batches: dict[str, np.ndarray] = {}
        for name, method_config in METHOD_CONFIGS.items():
            environment = build_environment(SCENARIO)
            algorithm = build_algorithm(name, method_config, SCENARIO, seed)
            rng = np.random.default_rng(seed)
            batches[name] = environment.sample(
                algorithm.policy, SCENARIO["group_size"], rng
            )

        reference_name, reference_batch = next(iter(batches.items()))
        for name, batch in batches.items():
            np.testing.assert_array_equal(
                batch,
                reference_batch,
                err_msg=(
                    f"{name} drew a different first trajectory batch than "
                    f"{reference_name} at the same seed; the Pareto V1 pairing "
                    "assumption does not hold for this method/step-size pair."
                ),
            )

    def test_different_seeds_are_not_accidentally_identical(self) -> None:
        """Sanity check for the test above: distinct seeds must actually
        produce distinct batches, otherwise the equality check is vacuous."""
        environment = build_environment(SCENARIO)
        method_config = METHOD_CONFIGS["uniform_eta100"]
        algorithm_a = build_algorithm("uniform_eta100", method_config, SCENARIO, 7)
        algorithm_b = build_algorithm("uniform_eta100", method_config, SCENARIO, 8)
        batch_a = environment.sample(
            algorithm_a.policy, SCENARIO["group_size"], np.random.default_rng(7)
        )
        batch_b = environment.sample(
            algorithm_b.policy, SCENARIO["group_size"], np.random.default_rng(8)
        )
        self.assertFalse(np.array_equal(batch_a, batch_b))

    def test_summary_per_seed_arrays_are_paired_across_methods(self) -> None:
        """`summarize()`'s per-seed arrays (added for Pareto V1) must list the
        same seeds in the same order for every method in a scenario, since the
        evaluator zips them positionally to form d_AUC(s) and r_KL(s)."""
        rows: list[dict[str, object]] = []
        run_metadata: dict[tuple[str, str, int], dict[str, float]] = {}
        selected = ["uniform_eta050", "uniform_eta150", "online_eta050", "online_eta150"]
        seeds = [3, 1, 4]
        for name in selected:
            method_config = METHOD_CONFIGS[name]
            for seed in seeds:
                run_rows, metadata = run_one(SCENARIO, name, method_config, seed, 3)
                rows.extend(run_rows)
                run_metadata[(SCENARIO["name"], name, seed)] = metadata

        summary = summarize(rows, run_metadata)
        scenario_summary = summary[SCENARIO["name"]]
        reference_seeds = scenario_summary[selected[0]]["seeds"]
        self.assertEqual(reference_seeds, sorted(seeds))
        for name in selected:
            self.assertEqual(
                scenario_summary[name]["seeds"],
                reference_seeds,
                f"{name} has a different seed order than {selected[0]}",
            )
            self.assertEqual(
                len(scenario_summary[name]["success_auc_per_seed"]),
                len(reference_seeds),
            )
            self.assertEqual(
                len(scenario_summary[name]["cumulative_distractor_kl_per_seed"]),
                len(reference_seeds),
            )


if __name__ == "__main__":
    unittest.main()
