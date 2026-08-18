"""Build the `geometry-v2-2026-08-18` golden record.

Mirrors the schema of `paper/frozen/geometry-v1-2026-08-15.json` so the frozen
records stay comparable. Every provenance field is derived from the repository
and the run outputs; nothing is typed in by hand except the narrative fields,
which are supplied on the command line.

Usage:
    python experiments/archive_geometry_v2.py \
        --results results/geometry_v2 \
        --evaluation results/geometry_v2/protocol_evaluation.json \
        --headline "..." --provenance "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PROTOCOL_ID = "geometry-v2-2026-08-18"

# Fixed before execution so the record cannot be softened after seeing results.
INTERPRETATION_LIMITS = [
    "Every result rests on the softmax-linear one-hot tie-group "
    "parameterization of assumption (M1). The collapse identity, the "
    "convexity argument, and the exact projection all depend on it. Nothing "
    "here transfers to non-linear function approximation.",
    "Lemma 2 supplies a NECESSARY structural condition, not a sufficient one. "
    "In particular the observed null on partial_feature_aliasing is not "
    "explained by the theory: that scenario contains a purely-distractor "
    "tie-group whose down-weighting should reduce distractor KL at zero "
    "critical cost. The mean-normalization cancellation conjecture is not "
    "assumed and not tested by this protocol.",
    "The five inherited scenarios are seen data. Only the three A3 arms are "
    "new under this protocol.",
    "lambda*=3.0 and mu*=0.0 remain frozen from geometry-v1 and are not "
    "re-selected here. This protocol says nothing about their optimality.",
    "The heterogeneous discriminator arm cannot match the per-group "
    "critical-count multiset of the other two arms, because heterogeneous "
    "finite yield ratios at a fixed group-size multiset require unequal "
    "critical counts. Its evidence is correspondingly weaker than the "
    "exactly-matched homogeneous/pure-critical pair.",
    "No runtime interval may be quoted; wall-clock ratios do not reproduce "
    "across machines. See docs/ERRATA.md E7.",
]


def sha256_of(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def environment() -> dict[str, str]:
    import matplotlib
    import numpy

    return OrderedDict(
        [
            ("python", platform.python_version()),
            ("numpy", numpy.__version__),
            ("matplotlib", matplotlib.__version__),
            ("platform", platform.platform()),
            ("processor", platform.processor()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--headline", required=True)
    parser.add_argument("--mechanism", default="")
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--coverage-recheck", default="")
    arguments = parser.parse_args()

    if git("status", "--porcelain"):
        sys.exit(
            "working tree is dirty; the golden record must record a clean "
            "execution commit"
        )

    evaluation = json.loads(arguments.evaluation.read_text(encoding="utf-8"))
    if evaluation.get("protocol_id") != PROTOCOL_ID:
        sys.exit(f"evaluation is not {PROTOCOL_ID}")

    config_path = REPO / "configs" / "geometry_v2.json"
    generated: "OrderedDict[str, Any]" = OrderedDict()
    generated["geometry_v2.json"] = sha256_of(config_path)
    for name in sorted(p.name for p in arguments.results.iterdir() if p.is_file()):
        generated[name] = sha256_of(arguments.results / name)

    record = OrderedDict(
        [
            ("record_type", "golden_record"),
            ("record_status", "ORIGINAL"),
            ("protocol_id", PROTOCOL_ID),
            ("phase", "test"),
            ("protocol_commit", git("rev-list", "-1", "HEAD",
                                    "--", "docs/GEOMETRY_V2_PROTOCOL.md")),
            ("structure_freeze_commit", git("rev-list", "-1", "HEAD",
                                            "--", "configs/geometry_v2.json")),
            ("execution_commit", git("rev-parse", "HEAD")),
            ("executed_on", date.today().isoformat()),
            ("provenance", arguments.provenance),
            ("coverage_recheck", arguments.coverage_recheck),
            ("environment", environment()),
            ("config", OrderedDict(
                [("path", "configs/geometry_v2.json")] +
                list(sha256_of(config_path).items())
            )),
            ("hypotheses", OrderedDict([
                ("h_pure", "A pure-critical tie-group is required for a "
                           "frontier advantage."),
                ("h_hetero", "Yield-ratio heterogeneity is required, and "
                             "pure_crit >= 1 is its extreme case."),
                ("discriminating_scenario",
                 evaluation["discriminating_scenario"]),
                ("discriminating_verdict",
                 evaluation["discriminating_verdict"]),
            ])),
            ("decision", OrderedDict([
                ("outcome", evaluation["outcome"]),
                ("verdicts", evaluation["verdicts"]),
                ("h_pure_satisfied", evaluation["hypothesis_h_pure_satisfied"]),
                ("h_hetero_satisfied",
                 evaluation["hypothesis_h_hetero_satisfied"]),
                ("control_1_invariant_holds",
                 evaluation["control_1_invariant"]["holds"]),
                ("control_2_no_spurious_pass_holds",
                 evaluation["control_2_no_spurious_pass"]["holds"]),
            ])),
            ("headline", arguments.headline),
            ("mechanism", arguments.mechanism),
            ("interpretation_limits", INTERPRETATION_LIMITS),
            ("generated_files", generated),
            ("protocol_evaluation_archive",
             f"paper/frozen/{PROTOCOL_ID}-evaluation.json"),
            ("scenario_results", evaluation["scenario_results"]),
        ]
    )

    out = REPO / "paper" / "frozen" / f"{PROTOCOL_ID}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    archive = REPO / "paper" / "frozen" / f"{PROTOCOL_ID}-evaluation.json"
    archive.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {archive}")
    print(f"outcome: {evaluation['outcome']}")


if __name__ == "__main__":
    main()
