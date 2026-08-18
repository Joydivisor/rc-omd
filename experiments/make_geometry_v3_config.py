"""Generate `configs/geometry_v3.json`: the scenario family and frozen split.

Runs the learnability screen during generation, so every family member is
known-learnable before the split is drawn. Nothing here inspects the frontier
metric; the split is fixed by a committed seed and no scenario is evaluated.

Usage:  python experiments/make_geometry_v3_config.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from experiments.geometry_v3_family import (
    EVALUATION_SEEDS,
    FAMILY_SIZE,
    GENERATION_SEED,
    LEARNABILITY_ETA,
    LEARNABILITY_SEEDS,
    LEARNABILITY_THRESHOLD,
    PROTOCOL_ID,
    build_scenario,
    forbidden_profiles,
    predictor_values,
    profile_key,
    sample_profile,
    split_family,
)
from experiments.geometry_v2_scenarios import is_realizable, structure
from experiments.run_reliability_diagnostics import run_one

REPO = Path(__file__).resolve().parent.parent


def learnable(scenario: dict) -> tuple[bool, float]:
    """Uniform baseline must gain at least the frozen threshold."""

    method_config = {
        "algorithm": "projected_group_omd",
        "step_size": LEARNABILITY_ETA,
        "projection_ridge": 0.0,
        "projection_steps": 60,
        "projection_learning_rate": 0.5,
        "projection_tolerance": 1e-9,
    }
    gains = []
    for seed in LEARNABILITY_SEEDS:
        history, _ = run_one(scenario, "probe", method_config, seed, 200)
        gains.append(
            history[-1]["success_probability"] - history[0]["success_probability"]
        )
    gain = float(np.mean(gains))
    return gain >= LEARNABILITY_THRESHOLD, gain


def main() -> None:
    v1 = json.loads((REPO / "configs" / "geometry_v1.json").read_text("utf-8"))
    v2 = json.loads((REPO / "configs" / "geometry_v2.json").read_text("utf-8"))
    banned = forbidden_profiles([v1, v2])

    rng = np.random.default_rng(GENERATION_SEED)
    scenarios: list[dict] = []
    gains: dict[str, float] = {}
    seen: set[tuple] = set()
    attempts = 0

    while len(scenarios) < FAMILY_SIZE:
        attempts += 1
        if attempts > 100000:
            raise SystemExit("could not fill the family under the frozen constraints")
        profile = sample_profile(rng)
        if profile is None:
            continue
        key = profile_key(profile)
        if key in banned or key in seen:
            continue
        name = f"gv3_{len(scenarios):03d}"
        scenario = build_scenario(name, profile)
        if not is_realizable(scenario):
            continue
        ok, gain = learnable(scenario)
        if not ok:
            continue
        seen.add(key)
        gains[name] = gain
        scenarios.append(scenario)
        print(f"  accepted {name} H={scenario['horizon']:>2} "
              f"groups={len(profile)} gain={gain:+.4f}")

    names = [s["name"] for s in scenarios]
    split = split_family(names)

    # Designated invariant control: every tie-group has c(g) == d(g), so
    # alpha == 1. Held outside the family and outside the split; it contributes
    # to no correlation. See the Control 1 amendment in the protocol.
    control = build_scenario("gv3_control_complete", ((1, 1), (2, 2), (3, 3)))
    control_alpha = structure(control)["alpha"]
    if control_alpha != 1.0:
        raise SystemExit(f"control alpha is {control_alpha}, expected 1.0")
    scenarios.append(control)

    methods = OrderedDict(v2["methods"])  # inherited unchanged

    config = OrderedDict(
        [
            ("protocol_id", PROTOCOL_ID),
            ("phase", "family"),
            (
                "generated_from",
                "docs/GEOMETRY_V3_PROTOCOL.md; family by rejection sampling at "
                f"seed {GENERATION_SEED}; split at seed 20260819; methods and "
                "grid inherited verbatim from configs/geometry_v2.json",
            ),
            ("generation", OrderedDict([
                ("attempts", attempts),
                ("family_size", len(scenarios)),
                ("learnability_threshold", LEARNABILITY_THRESHOLD),
                ("learnability_seeds", list(LEARNABILITY_SEEDS)),
                ("baseline_gain", gains),
            ])),
            ("split", split),
            ("control_scenario", "gv3_control_complete"),
            ("scenario_structure",
             OrderedDict((s["name"], structure(s)) for s in scenarios)),
            ("predictors",
             OrderedDict((s["name"], predictor_values(s)) for s in scenarios)),
            ("scenarios", scenarios),
            ("methods", methods),
            ("training", OrderedDict([
                ("evaluation_interval", 5),
                ("seeds", list(EVALUATION_SEEDS)),
            ])),
            ("decision_rule", OrderedDict([
                ("candidate_method", "rwp_eta125_lam3"),
                ("grid_prefix", "uniform_eta"),
                ("max_dropped_seeds", 2),
                ("selection_threshold_abs_rho", 0.70),
                ("confirmation_rho_lower_bound", 0.50),
                ("bootstrap_resamples", 10000),
                ("bootstrap_seed", 20260820),
                ("invariant_tolerance", 1e-9),
            ])),
            ("output_directory", "results/geometry_v3"),
        ]
    )

    out = REPO / "configs" / "geometry_v3.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = len(scenarios) * len(methods) * len(EVALUATION_SEEDS)
    print(f"\nwrote {out}")
    print(f"family {len(scenarios)} | discovery {len(split['discovery'])} | "
          f"confirmation {len(split['confirmation'])}")
    print(f"methods {len(methods)} | seeds {len(EVALUATION_SEEDS)} | runs {runs}")
    alphas = [structure(s)["alpha"] for s in scenarios]
    print(f"alpha range {min(alphas):.3f} to {max(alphas):.3f}; "
          f"complete-aliasing members: {sum(1 for a in alphas if a == 1.0)} "
          "(the control, held outside the family)")


if __name__ == "__main__":
    main()
