"""Scenario generation and structural statistics for `geometry-v2-2026-08-18`.

Implements the generation rule and the structural definitions of
`docs/GEOMETRY_V2_PROTOCOL.md` verbatim. Nothing here may introduce or relax a
rule stated in that document.

The A3 triple is three scenarios matched on horizon, action count, aliasing
index, critical and distractor counts, tie-group count, and tie-group size
multiset. They differ only in the structure this protocol tests: whether a
critical-bearing tie-group has zero distractors (`pure_crit`), and whether the
yield ratios `c(g)/d(g)` of critical-bearing groups are homogeneous.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Any, Iterable

# --- frozen scenario constants -------------------------------------------

HORIZON = 12
N_ACTIONS = 3
MINIMUM_MATCHES = 2
ITERATIONS = 400
GROUP_SIZE = 48

# Tie-group profiles as ordered (critical_count, distractor_count) pairs.
# Emission order is part of the frozen generation rule: groups are laid out in
# this order and positions are assigned in ascending index order.
A3_PROFILES: "OrderedDict[str, tuple[tuple[int, int], ...]]" = OrderedDict(
    [
        ("geom_v2_a3_pure0_homog", ((0, 2), (0, 4), (2, 1), (2, 1))),
        ("geom_v2_a3_pure0_hetero", ((0, 2), (0, 3), (1, 2), (3, 1))),
        ("geom_v2_a3_pure_ge1", ((0, 3), (0, 3), (2, 0), (2, 2))),
    ]
)

# Properties the A3 arms must agree on. `critical_counts` is deliberately not
# universal: the heterogeneous arm cannot match it, because heterogeneous
# finite yield ratios at fixed group sizes require unequal critical counts.
MATCHED_PROPERTIES = (
    "horizon",
    "n_actions",
    "alpha",
    "n_critical",
    "n_distractor",
    "n_groups",
    "group_sizes",
)


# --- generation rule ------------------------------------------------------


def build_scenario(name: str, profile: Iterable[tuple[int, int]]) -> dict[str, Any]:
    """Emit one scenario from a tie-group profile.

    Generation rule, frozen:

    1. Tie-groups are emitted in profile order; positions are assigned to
       groups in ascending index order.
    2. Features are one-hot indicators of tie-group membership.
    3. Within each group, critical positions precede distractor positions.
    4. Critical-bearing groups are ranked by first position; the group of rank
       ``j`` assigns target action ``j % n_actions`` to all of its critical
       positions, which guarantees within-group target consistency and hence
       realizability.
    """

    profile = tuple(profile)
    horizon = sum(c + d for c, d in profile)
    if horizon != HORIZON:
        raise ValueError(f"{name}: profile covers {horizon} positions, need {HORIZON}")

    features = [[0] * len(profile) for _ in range(HORIZON)]
    critical_positions: list[int] = []
    critical_by_group: list[list[int]] = []

    position = 0
    for column, (n_crit, n_distr) in enumerate(profile):
        group_critical: list[int] = []
        for _ in range(n_crit):
            features[position][column] = 1
            critical_positions.append(position)
            group_critical.append(position)
            position += 1
        for _ in range(n_distr):
            features[position][column] = 1
            position += 1
        critical_by_group.append(group_critical)

    # Rank by first position; groups are already emitted in position order, so
    # filtering preserves that ranking.
    target_of_position: dict[int, int] = {}
    rank = 0
    for group_critical in critical_by_group:
        if not group_critical:
            continue
        action = rank % N_ACTIONS
        for member in group_critical:
            target_of_position[member] = action
        rank += 1

    return OrderedDict(
        [
            ("name", name),
            ("environment", "threshold"),
            ("horizon", HORIZON),
            ("n_actions", N_ACTIONS),
            ("critical_positions", critical_positions),
            ("target_actions", [target_of_position[k] for k in critical_positions]),
            ("minimum_matches", MINIMUM_MATCHES),
            (
                "initial_policy",
                [[1.0 / N_ACTIONS] * N_ACTIONS for _ in range(HORIZON)],
            ),
            ("features", features),
            ("iterations", ITERATIONS),
            ("group_size", GROUP_SIZE),
        ]
    )


def build_a3_triple() -> list[dict[str, Any]]:
    """The three A3 arms, in frozen order."""

    return [build_scenario(name, profile) for name, profile in A3_PROFILES.items()]


# --- structural statistics ------------------------------------------------


def tie_groups(scenario: dict[str, Any]) -> list[list[int]]:
    """Positions grouped by identical feature row, ordered by first member."""

    features = scenario["features"]
    groups: "OrderedDict[tuple[int, ...], list[int]]" = OrderedDict()
    for position in range(int(scenario["horizon"])):
        groups.setdefault(tuple(features[position]), []).append(position)
    return list(groups.values())


def structure(scenario: dict[str, Any]) -> dict[str, Any]:
    """Every structural quantity the protocol defines, for one scenario."""

    horizon = int(scenario["horizon"])
    critical = set(scenario["critical_positions"])
    groups = tie_groups(scenario)

    counts = [
        (
            sum(1 for k in members if k in critical),
            sum(1 for k in members if k not in critical),
        )
        for members in groups
    ]
    spread = sum(abs(c - d) for c, d in counts)

    # Yield ratio c/d of each critical-bearing group; None encodes +inf.
    ratios: list[Fraction | None] = [
        (Fraction(c, d) if d > 0 else None) for c, d in counts if c > 0
    ]
    distinct = {("inf" if r is None else r) for r in ratios}
    if len(distinct) <= 1:
        ratio_class = "homogeneous"
    else:
        ratio_class = "heterogeneous"

    return {
        "name": scenario.get("name"),
        "horizon": horizon,
        "n_actions": int(scenario["n_actions"]),
        "n_critical": len(critical),
        "n_distractor": horizon - len(critical),
        "n_groups": len(groups),
        "group_sizes": sorted(len(members) for members in groups),
        "group_counts": sorted(counts),
        "critical_counts": sorted(c for c, _ in counts if c > 0),
        "spread": spread,
        "alpha": 1.0 - spread / horizon,
        "pure_crit": sum(1 for c, d in counts if c > 0 and d == 0),
        "yield_ratios": ["inf" if r is None else str(r) for r in ratios],
        "ratio_class": ratio_class,
    }


def alpha_is_admissible(horizon: int, alpha: float) -> bool:
    """Lemma 1: `spread == horizon (mod 2)`, so alpha is quantized by 2/H.

    `alpha = 1/2` therefore requires `4 | horizon`; it is unreachable at
    `horizon = 10`.
    """

    spread = (1.0 - alpha) * horizon
    nearest = round(spread)
    if abs(spread - nearest) > 1e-9:
        return False
    return nearest % 2 == horizon % 2


def is_realizable(scenario: dict[str, Any]) -> bool:
    """No tie-group may hold critical positions with conflicting targets.

    A conflict makes the optimum unrepresentable under the shared-feature
    parameterization; `geom_dev_separable` failed this once and had to be
    redesigned.
    """

    targets = dict(
        zip(scenario["critical_positions"], scenario["target_actions"], strict=True)
    )
    for members in tie_groups(scenario):
        actions = {targets[k] for k in members if k in targets}
        if len(actions) > 1:
            return False
    return True


def validate_a3_triple(scenarios: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check the A3 arms agree on every property the protocol requires.

    Returns a report rather than raising, so callers can archive it.
    """

    structures = [structure(scenario) for scenario in scenarios]
    agreement = {
        prop: len({_hashable(s[prop]) for s in structures}) == 1
        for prop in MATCHED_PROPERTIES
    }
    return {
        "arms": structures,
        "matched_properties": agreement,
        "all_matched": all(agreement.values()),
        "realizable": all(
            is_realizable(scenario) for scenario in scenarios  # type: ignore[arg-type]
        ),
        # The exactly-matched pair: these differ only in pure_crit.
        "exact_pair_matched": (
            _pair_matches(structures, "geom_v2_a3_pure0_homog", "geom_v2_a3_pure_ge1")
        ),
        "pure_crit": {s["name"]: s["pure_crit"] for s in structures},
        "ratio_class": {s["name"]: s["ratio_class"] for s in structures},
    }


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _pair_matches(structures: list[dict[str, Any]], left: str, right: str) -> bool:
    """The homogeneous and pure-critical arms additionally match critical counts."""

    lookup = {s["name"]: s for s in structures}
    if left not in lookup or right not in lookup:
        return False
    a, b = lookup[left], lookup[right]
    same = all(
        _hashable(a[prop]) == _hashable(b[prop])
        for prop in MATCHED_PROPERTIES + ("critical_counts",)
    )
    return same and a["pure_crit"] != b["pure_crit"]
