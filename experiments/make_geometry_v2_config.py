"""Generate `configs/geometry_v2.json` from the frozen protocol.

Implements `docs/GEOMETRY_V2_PROTOCOL.md` verbatim: the three A3 arms are
generated from their tie-group profiles, the five inherited scenarios are
copied byte-for-byte from `configs/geometry_v1.json` so they cannot drift, and
every grid value, hyperparameter, seed, and threshold is taken from the
protocol without re-selection.

Usage:  python experiments/make_geometry_v2_config.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from experiments.geometry_v2_scenarios import (
    build_a3_triple,
    structure,
    validate_a3_triple,
)

PROTOCOL_ID = "geometry-v2-2026-08-18"
REPO = Path(__file__).resolve().parent.parent

# Frozen 18-point grid, floor 0.025. The floor is set by measurement: the worst
# development seed required eta = 0.0330 for the grid to bracket RWP-OMD, and a
# floor of 0.05 dropped 19 of 40 seeds on geom_dev_separable.
ETA_GRID = (
    0.025, 0.03125, 0.0375, 0.05, 0.0625, 0.075, 0.10, 0.125, 0.15,
    0.20, 0.25, 0.30, 0.40, 0.50, 0.625, 0.75, 1.00, 1.25,
)

# Scenarios inherited verbatim from geometry-v1, with their protocol roles.
INHERITED = OrderedDict(
    [
        ("separable_shared_features", "positive_anchor"),
        ("geom_holdout_a", "positive_anchor"),
        ("geom_holdout_b", "positive_anchor"),
        ("partial_feature_aliasing", "null_anchor"),
        ("complete_feature_aliasing_negative_control", "invariant_control"),
    ]
)

A3_ROLES = OrderedDict(
    [
        ("geom_v2_a3_pure_ge1", "a3_positive"),
        ("geom_v2_a3_pure0_homog", "a3_null"),
        ("geom_v2_a3_pure0_hetero", "a3_discriminator"),
    ]
)

PROJECTION = OrderedDict(
    [
        ("projection_steps", 60),
        ("projection_learning_rate", 0.5),
        ("projection_tolerance", 1e-9),
    ]
)
RELIABILITY = OrderedDict(
    [
        ("reliability_decay", 0.9),
        ("confidence_multiplier", 1.0),
        ("warmup_effective_samples", 8.0),
        ("reliability_floor", 0.1),
    ]
)


def grid_method_name(eta: float) -> str:
    return f"uniform_eta{int(round(eta * 100000)):06d}"


def main() -> None:
    v1 = json.loads((REPO / "configs" / "geometry_v1.json").read_text("utf-8"))
    v1_scenarios = {s["name"]: s for s in v1["scenarios"]}

    triple = build_a3_triple()
    report = validate_a3_triple(triple)
    if not report["all_matched"]:
        raise SystemExit(f"A3 arms are not matched: {report['matched_properties']}")
    if not report["realizable"]:
        raise SystemExit("A3 arms contain a tie-group with conflicting targets")
    if not report["exact_pair_matched"]:
        raise SystemExit("homog/pure_ge1 pair is not exactly matched")

    scenarios: list[dict] = []
    roles: "OrderedDict[str, str]" = OrderedDict()
    by_name = {s["name"]: s for s in triple}
    for name, role in A3_ROLES.items():
        scenarios.append(by_name[name])
        roles[name] = role
    for name, role in INHERITED.items():
        if name not in v1_scenarios:
            raise SystemExit(f"inherited scenario {name} missing from geometry_v1")
        scenarios.append(v1_scenarios[name])
        roles[name] = role

    methods: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for eta in ETA_GRID:
        methods[grid_method_name(eta)] = OrderedDict(
            [("algorithm", "projected_group_omd"), ("step_size", eta),
             ("projection_ridge", 0.0)] + list(PROJECTION.items())
        )
    methods["rwp_eta125_lam3"] = OrderedDict(
        [("algorithm", "rwp_omd"), ("step_size", 1.25),
         ("projection_lambda", 3.0), ("projection_ridge", 0.0)]
        + list(PROJECTION.items()) + list(RELIABILITY.items())
    )
    methods["projected_online_eta125"] = OrderedDict(
        [("algorithm", "projected_online_rc_omd"), ("step_size", 1.25),
         ("projection_ridge", 0.0)]
        + list(PROJECTION.items()) + list(RELIABILITY.items())
    )

    config = OrderedDict(
        [
            ("protocol_id", PROTOCOL_ID),
            ("phase", "test"),
            (
                "generated_from",
                "docs/GEOMETRY_V2_PROTOCOL.md; A3 arms from "
                "experiments/geometry_v2_scenarios.py; the five inherited "
                "scenarios copied verbatim from configs/geometry_v1.json",
            ),
            ("scenario_structure",
             OrderedDict((s["name"], structure(s)) for s in scenarios)),
            ("scenario_roles", roles),
            ("a3_validation", report["matched_properties"]),
            ("scenarios", scenarios),
            ("methods", methods),
            ("training", OrderedDict([
                ("evaluation_interval", 5),
                ("seeds", list(range(500, 520))),
            ])),
            ("decision_rule", OrderedDict([
                ("candidate_method", "rwp_eta125_lam3"),
                ("v1_reference_method", "projected_online_eta125"),
                ("grid_prefix", "uniform_eta"),
                ("eta_grid", list(ETA_GRID)),
                ("max_dropped_seeds", 2),
                ("invariant_scenario",
                 "complete_feature_aliasing_negative_control"),
                ("invariant_tolerance", 1e-9),
                # Pre-registered predictions. The discriminator is the only
                # scenario on which the two hypotheses disagree.
                ("predictions_h_pure", OrderedDict([
                    ("geom_v2_a3_pure_ge1", "positive"),
                    ("geom_v2_a3_pure0_homog", "null"),
                    ("geom_v2_a3_pure0_hetero", "null"),
                    ("separable_shared_features", "positive"),
                    ("geom_holdout_a", "positive"),
                    ("geom_holdout_b", "positive"),
                    ("partial_feature_aliasing", "null"),
                    ("complete_feature_aliasing_negative_control",
                     "null_or_negative"),
                ])),
                ("predictions_h_hetero", OrderedDict([
                    ("geom_v2_a3_pure_ge1", "positive"),
                    ("geom_v2_a3_pure0_homog", "null"),
                    ("geom_v2_a3_pure0_hetero", "positive"),
                    ("separable_shared_features", "positive"),
                    ("geom_holdout_a", "positive"),
                    ("geom_holdout_b", "positive"),
                    ("partial_feature_aliasing", "null"),
                    ("complete_feature_aliasing_negative_control",
                     "null_or_negative"),
                ])),
                ("discriminating_scenario", "geom_v2_a3_pure0_hetero"),
                ("no_spurious_pass_scenarios", [
                    "geom_v2_a3_pure0_homog",
                    "partial_feature_aliasing",
                ]),
            ])),
            ("output_directory", "results/geometry_v2"),
        ]
    )

    out = REPO / "configs" / "geometry_v2.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = len(scenarios) * len(methods) * 20
    print(f"wrote {out}")
    print(f"scenarios {len(scenarios)} | methods {len(methods)} | seeds 20 | runs {runs}")
    for name, role in roles.items():
        info = structure(config["scenarios"][[s["name"] for s in scenarios].index(name)])
        print(f"  {name:<44}{role:<18}alpha={info['alpha']:.4f} "
              f"pure_crit={info['pure_crit']} {info['ratio_class']}")


if __name__ == "__main__":
    main()
