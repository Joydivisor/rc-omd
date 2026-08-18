"""Generate `configs/stage_b_mlp.json`: instances, noise, and the frozen split.

24 base scenarios are sampled under the geometry-v3 constraints, with no
tie-group profile reused from geometry-v1, -v2, or -v3, and each is instantiated
at three feature-noise levels. The split is by **base scenario**, so the same
structure cannot appear on both sides of the split at different noise levels.

Nothing here measures the frontier metric or `alpha`; the split is fixed by a
committed seed before any instance runs.

Usage:  python experiments/make_stage_b_config.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from experiments.geometry_v2_scenarios import is_realizable, structure
from experiments.geometry_v3_family import (
    LEARNABILITY_THRESHOLD,
    build_scenario,
    forbidden_profiles,
    profile_key,
    sample_profile,
)
from experiments.make_geometry_v3_config import learnable

REPO = Path(__file__).resolve().parent.parent
PROTOCOL_ID = "stage-b-mlp-2026-08-19"

N_BASE = 24
N_DISCOVERY_BASE = 14
EPS_LEVELS = (0.0, 0.10, 0.30)

GENERATION_SEED = 20260821
NOISE_SEED = 20260821
SPLIT_SEED = 20260821
DISCOVERY_SEEDS = tuple(range(700, 720))
CONFIRMATION_SEEDS = tuple(range(800, 820))

MLP_HIDDEN = 32
CONTROL_PROFILE = ((1, 1), (2, 2), (3, 3))


def instance_name(base: str, eps: float) -> str:
    return f"{base}_eps{int(round(eps * 100)):03d}"


def perturb(features: list[list[int]], eps: float, rng) -> list[list[float]]:
    matrix = np.asarray(features, dtype=np.float64)
    if eps == 0.0:
        return matrix.tolist()
    return (matrix + eps * rng.normal(size=matrix.shape)).tolist()


def main() -> None:
    banned = set()
    for name in ("geometry_v1.json", "geometry_v2.json", "geometry_v3.json"):
        banned |= forbidden_profiles(
            [json.loads((REPO / "configs" / name).read_text("utf-8"))]
        )

    rng = np.random.default_rng(GENERATION_SEED)
    bases: list[dict] = []
    gains: dict[str, float] = {}
    seen: set[tuple] = set()
    attempts = 0
    while len(bases) < N_BASE:
        attempts += 1
        if attempts > 100000:
            raise SystemExit("could not fill the base set")
        profile = sample_profile(rng)
        if profile is None:
            continue
        key = profile_key(profile)
        if key in banned or key in seen:
            continue
        name = f"sb_{len(bases):03d}"
        scenario = build_scenario(name, profile)
        if not is_realizable(scenario):
            continue
        ok, gain = learnable(scenario)
        if not ok:
            continue
        seen.add(key)
        gains[name] = gain
        bases.append(scenario)
        print(f"  accepted {name} H={scenario['horizon']:>2} gain={gain:+.4f}")

    control = build_scenario("sb_control_complete", CONTROL_PROFILE)
    if structure(control)["alpha"] != 1.0:
        raise SystemExit("control is not complete aliasing")

    noise_rng = np.random.default_rng(NOISE_SEED)
    scenarios: list[dict] = []
    instance_of_base: dict[str, list[str]] = {}
    discrete_alpha: dict[str, float] = {}
    for scenario in bases + [control]:
        base_name = scenario["name"]
        instance_of_base[base_name] = []
        for eps in EPS_LEVELS:
            instance = OrderedDict(scenario)
            instance["name"] = instance_name(base_name, eps)
            instance["features"] = perturb(scenario["features"], eps, noise_rng)
            instance["base_scenario"] = base_name
            instance["feature_noise"] = eps
            scenarios.append(instance)
            instance_of_base[base_name].append(instance["name"])
            # alpha is defined on the UNPERTURBED structure; noise changes the
            # parameterization, never the task.
            discrete_alpha[instance["name"]] = structure(scenario)["alpha"]

    # Per-instance seed assignment: discovery and confirmation use disjoint
    # seed sets, so an instance must carry its own rather than the union.
    split_rng_pre = np.random.default_rng(SPLIT_SEED)
    order = split_rng_pre.permutation(len(bases))
    discovery_bases = sorted(bases[i]["name"] for i in order[:N_DISCOVERY_BASE])
    confirmation_bases = sorted(bases[i]["name"] for i in order[N_DISCOVERY_BASE:])
    split = OrderedDict([
        ("discovery_bases", discovery_bases),
        ("confirmation_bases", confirmation_bases),
        ("discovery", sorted(
            n for b in discovery_bases for n in instance_of_base[b])),
        ("confirmation", sorted(
            n for b in confirmation_bases for n in instance_of_base[b])),
    ])

    confirmation_set = set(split["confirmation"])
    for instance in scenarios:
        instance["seeds"] = list(
            CONFIRMATION_SEEDS if instance["name"] in confirmation_set
            else DISCOVERY_SEEDS
        )

    v2 = json.loads((REPO / "configs" / "geometry_v2.json").read_text("utf-8"))
    methods: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for name, spec in v2["methods"].items():
        if not name.startswith("uniform_eta"):
            continue
        entry = OrderedDict(spec)
        entry["algorithm"] = "mlp_projected_group_omd"
        entry["mlp_hidden"] = MLP_HIDDEN
        methods[name] = entry
    rwp = OrderedDict(v2["methods"]["rwp_eta125_lam3"])
    rwp["algorithm"] = "mlp_rwp_omd"
    rwp["mlp_hidden"] = MLP_HIDDEN
    methods["rwp_eta125_lam3"] = rwp

    config = OrderedDict([
        ("protocol_id", PROTOCOL_ID),
        ("phase", "family"),
        ("generated_from",
         "docs/STAGE_B_MLP_PROTOCOL.md; base scenarios by rejection sampling "
         f"at seed {GENERATION_SEED} under the geometry-v3 constraints; noise "
         f"at seed {NOISE_SEED}; split by base scenario at seed {SPLIT_SEED}; "
         "eta grid inherited from configs/geometry_v2.json"),
        ("eps_levels", list(EPS_LEVELS)),
        ("mlp_hidden", MLP_HIDDEN),
        ("generation", OrderedDict([
            ("attempts", attempts),
            ("n_base", len(bases)),
            ("learnability_threshold", LEARNABILITY_THRESHOLD),
            ("baseline_gain", gains),
        ])),
        ("split", split),
        ("control_base", "sb_control_complete"),
        ("control_instances", instance_of_base["sb_control_complete"]),
        ("discrete_alpha", discrete_alpha),
        ("scenarios", scenarios),
        ("methods", methods),
        ("training", OrderedDict([
            ("evaluation_interval", 5),
            ("seeds", list(DISCOVERY_SEEDS) + list(CONFIRMATION_SEEDS)),
            ("discovery_seeds", list(DISCOVERY_SEEDS)),
            ("confirmation_seeds", list(CONFIRMATION_SEEDS)),
        ])),
        ("decision_rule", OrderedDict([
            ("candidate_method", "rwp_eta125_lam3"),
            ("baseline_alpha_method", "uniform_eta075000"),
            ("grid_prefix", "uniform_eta"),
            ("max_dropped_seeds", 2),
            ("calibration_tolerance", 1e-6),
            ("expected_sign", -1),
            ("confirmation_magnitude_bound", 0.50),
            ("bootstrap_resamples", 10000),
            ("bootstrap_seed", 20260822),
            ("invariant_tolerance", 1e-9),
        ])),
        ("output_directory", "results/stage_b_mlp"),
    ])

    out = REPO / "configs" / "stage_b_mlp.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"base {len(bases)} (+1 control) | instances {len(scenarios)} | "
          f"methods {len(methods)}")
    print(f"discovery {len(split['discovery'])} instances from "
          f"{len(discovery_bases)} bases | confirmation "
          f"{len(split['confirmation'])} from {len(confirmation_bases)}")
    print(f"runs {sum(len(s['seeds']) for s in scenarios) * len(methods)}")


if __name__ == "__main__":
    main()
