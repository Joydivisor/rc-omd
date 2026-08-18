"""Execute a geometry-v3 config scenario by scenario, with resume.

The shared runner accumulates every history row in memory and writes only at
the end, so an interruption loses the whole job. geometry-v3 is 51 scenarios x
20 methods x 20 seeds, long enough that this is a real risk -- one geometry-v2
execution was already lost that way.

This runner writes one summary shard per scenario under ``partial/`` and skips
shards that already exist, so a restart resumes where it stopped. Per-seed
arrays match the shared runner's ``summarize`` output exactly, so the resulting
``summary.json`` is interchangeable with one produced in a single pass.

Usage:
    python experiments/run_geometry_v3.py --config configs/geometry_v3.json
    python experiments/run_geometry_v3.py --config ... --only gv3_003 gv3_004
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.run_reliability_diagnostics import load_config, run_one, summarize


def run_scenario(
    scenario: dict[str, Any],
    methods: dict[str, Any],
    seeds: list[int],
    evaluation_interval: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metadata: dict[tuple[str, str, int], dict[str, float]] = {}
    for method, method_config in methods.items():
        for seed in seeds:
            run_rows, run_metadata = run_one(
                scenario, method, method_config, int(seed), evaluation_interval
            )
            rows.extend(run_rows)
            metadata[(scenario["name"], method, int(seed))] = run_metadata
    return summarize(rows, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    output = Path(config["output_directory"])
    shards = output / "partial"
    shards.mkdir(parents=True, exist_ok=True)

    methods = config["methods"]
    seeds = list(config["training"]["seeds"])
    interval = int(config["training"]["evaluation_interval"])

    selected = [
        s for s in config["scenarios"]
        if arguments.only is None or s["name"] in arguments.only
    ]
    for index, scenario in enumerate(selected, start=1):
        name = scenario["name"]
        shard = shards / f"{name}.json"
        if shard.exists():
            print(f"[{index}/{len(selected)}] {name}: shard present, skipping",
                  flush=True)
            continue
        # A scenario may carry its own seed list; stage-b uses disjoint seed
        # sets for discovery and confirmation instances.
        scenario_seeds = [int(s) for s in scenario.get("seeds", seeds)]
        print(f"[{index}/{len(selected)}] {name}: running "
              f"{len(methods)}x{len(scenario_seeds)} ...", flush=True)
        result = run_scenario(scenario, methods, scenario_seeds, interval)
        # Write to a temporary path first so an interrupted write cannot leave
        # a truncated shard that a later resume would trust.
        temporary = shard.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result[name]), encoding="utf-8")
        temporary.replace(shard)
        print(f"[{index}/{len(selected)}] {name}: done", flush=True)

    merged: dict[str, Any] = {}
    missing = []
    for scenario in config["scenarios"]:
        shard = shards / f"{scenario['name']}.json"
        if shard.exists():
            merged[scenario["name"]] = json.loads(shard.read_text(encoding="utf-8"))
        else:
            missing.append(scenario["name"])

    if missing:
        print(f"\n{len(missing)} scenario(s) still missing: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
        print("summary.json NOT written; re-run to resume")
        return

    (output / "summary.json").write_text(json.dumps(merged, indent=2),
                                         encoding="utf-8")
    print(f"\nall {len(merged)} scenarios complete; wrote {output/'summary.json'}")


if __name__ == "__main__":
    main()
