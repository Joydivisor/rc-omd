# Geometry V2 Protocol

Protocol ID: `geometry-v2-2026-08-18`

Status: **document frozen as of this commit.** The hypothesis, scenarios,
grids, statistics, matching rules, decision rules, and failure semantics below
are closed. They may not be altered under this protocol ID; a change requires a
new one and is post-hoc with respect to anything already observed.

Freezing completes in three stages, of which the first is done:

1. **Document frozen (this commit).** Protocol only. No code, no configs, no
   scenario files, no runs.
2. **Structure frozen.** Scenario generator, configs, evaluator, and tests are
   committed, implementing this document verbatim. Those artifacts may not
   introduce or relax any rule stated here; if implementation shows a rule to
   be wrong, it must be amended in an explicit commit *before* any sweep is
   run, never silently in code.
3. **Coverage confirmed.** The development re-check at the frozen `eta` floor
   is committed, demonstrating zero dropped seeds.

Only after stage 3 may any test scenario be executed.

The algorithm under test is unchanged and is specified in
[`PARAMETER_SPACE_GEOMETRY_DESIGN.md`](PARAMETER_SPACE_GEOMETRY_DESIGN.md).
`lambda* = 3.0` and `mu* = 0.0` remain frozen from `geometry-v1-2026-08-15` and
are **not** re-selected here.

## Purpose

`geometry-v1-2026-08-15` returned GO but held `eta` fixed, leaving the
step-size confound open. Post-hoc paired re-analysis of the frozen v1 test
seeds (recorded below) established two things:

- Under `partial_feature_aliasing` -- the scenario that produced the M6 NO-GO
  and that v1 was declared to have recovered -- RWP-OMD's advantage over a
  step-swept uniform frontier is **null**: `Delta_frontier = +0.00033`, 95% CI
  `[-0.00037, +0.00102]`. The allocation advantage over the v1 algorithm is
  also null: ratio `0.8185 [0.8039, 0.8333]` against v1's
  `0.8227 [0.8020, 0.8439]`.
- The two held-out scenarios do show a frontier advantage, and `alpha` does
  not predict which scenarios do.

This protocol asks one question:

> Is the presence of a **pure-critical tie-group** what determines whether
> RWP-OMD exceeds the step-size-swept uniform frontier?

## Definitions

Fix a scenario with horizon `H`, action count `A`, a set of critical positions,
and a feature matrix whose rows are one-hot indicators of tie-group membership.
Positions `k`, `k'` are in the same **tie-group** iff their feature rows are
equal. For tie-group `g` write `c(g)` for the number of critical positions in
`g` and `d(g)` for the number of distractor positions.

- `spread = sum_g |c(g) - d(g)|`
- **aliasing index** `alpha = 1 - spread / H`
- `g` is **pure-critical** iff `c(g) > 0` and `d(g) = 0`
- `pure_crit` = the number of pure-critical tie-groups
- a group is **critical-bearing** iff `c(g) > 0`
- the **yield ratio** of a critical-bearing group is `c(g)/d(g)`, taken as
  `+inf` when `d(g) = 0`
- a scenario is **ratio-homogeneous** iff all critical-bearing groups share one
  finite yield ratio, and **ratio-heterogeneous** iff at least two distinct
  finite yield ratios occur

## Theory

### Model assumptions

The results below hold under exactly these assumptions and are claimed under no
others.

- **(M1)** The policy is softmax-linear in one-hot tie-group features, so every
  position in a tie-group carries one shared policy row and any update moves
  them identically.
- **(M2)** Per-position movement is `D_KL(pi_new^k || pi_old^k)`; cumulative
  critical and distractor KL are sums of this quantity over the positions of
  the respective class.
- **(M3)** The update applies a per-group scalar weight to a common direction
  (`w_tilde_g` for RWP-OMD, `w_tilde_g == 1` for Uniform), with `eta` scaling
  all groups jointly.

### Lemma 1 (parity quantization of alpha)

`|c - d| = (c + d) mod 2`, so `spread = H mod 2` and therefore
`alpha` is confined to `{1 - 2j/H : j = 0, 1, ...}`. In particular `alpha = 1/2`
is unreachable unless `4 | H`, which is why the A3 arms below cannot be built at
`H = 10`.

### Lemma 2 (reachable cone)

Let `delta_g = D_KL(p'_g || p_g) >= 0` be the movement of group `g` under any
update. By (M1)-(M2),

```
CritKL = sum_g c(g) * delta_g          DistKL = sum_g d(g) * delta_g
```

Both are linear in `delta` with non-negative coefficients. Hence:

- **(a)** If `pure_crit = 0`, every critical-bearing group has `d(g) >= 1`, so
  `DistKL >= sum_{c(g)>0} delta_g >= CritKL / c_max` with
  `c_max = max_g c(g)`. The reachable set lies in a cone **bounded away from
  the critical axis**: no update produces critical movement without
  proportional distractor movement.
- **(b)** If `pure_crit >= 1`, an update supported on a pure-critical group
  `g0` gives `CritKL = c(g0) * delta > 0` and `DistKL = 0`. The critical axis
  itself is reachable.

### Corollary (rigidity under ratio homogeneity)

If the scenario is additionally ratio-homogeneous with common ratio `rho`, then
for any `delta` supported on critical-bearing groups, `DistKL = CritKL / rho`
**exactly**. Re-weighting among critical-bearing groups therefore cannot change
`DistKL` at fixed `CritKL`. The only remaining freedom is (i) overall scale and
(ii) movement of purely-distractor groups.

### What is NOT proved

Stated explicitly so it cannot be quietly assumed later.

Freedom (i) is exactly what the `eta` sweep traverses, so it confers no
frontier advantage. Freedom (ii) is **not** controlled by the corollary:
down-weighting a purely-distractor group reduces `DistKL` at zero `CritKL`
cost and would, on its face, produce a strict frontier advantage.
`partial_feature_aliasing` contains a `(c=0, d=3)` group, yet its measured
`Delta_frontier` is null. **The theory does not explain that null.** One
candidate mechanism -- that mean-normalization `w_tilde = w / mean(w)`
reallocates the removed mass onto the remaining groups and cancels the gain --
is a conjecture, is not assumed anywhere in this protocol, and is not tested by
it.

Consequently Lemma 2 supplies a **necessary** structural condition, not a
sufficient one. The empirical question this protocol settles is whether that
necessary condition is also the operative one.

## The confound this protocol exists to resolve

Across the nine scenario-points measured so far, `pure_crit >= 1` predicted the
sign of `Delta_frontier` in 9 of 9 cases while `alpha` did not. However:

| observed class | `pure_crit` | ratio structure | `Delta_frontier` |
|---|---:|---|---|
| `separable_shared_features`, `geom_dev_separable` | 3, 2 | all `+inf` | strongly positive |
| `geom_holdout_a`, `geom_holdout_b`, `geom_dev_partial_040`, `geom_dev_partial_080` | 1-2 | mixed `inf` + finite | positive |
| `partial_feature_aliasing`, `complete_*`, `geom_dev_complete` | 0 | homogeneous | null |

**Every scenario with `pure_crit = 0` measured so far is also
ratio-homogeneous.** No scenario has ever been run with `pure_crit = 0` and
heterogeneous finite yield ratios. The 9-of-9 record is therefore equally
consistent with two hypotheses:

- **H_pure**: a pure-critical group is required.
- **H_hetero**: yield-ratio heterogeneity is required, and `pure_crit >= 1` is
  merely its extreme case (`+inf` differs from any finite ratio).

Ratio heterogeneity is *not* sufficient on its own -- the separable scenarios
are ratio-homogeneous (all `+inf`) and strongly positive -- so H_hetero must be
read as "heterogeneity among yield ratios, with `+inf` admitted as a value".
The two hypotheses are distinguished by exactly one untested cell, which the A3
triple below supplies.

## Scenario set (frozen)

### A3 matched triple

`H = 12`, `A = 3`, `n_critical = 4`, `n_distractor = 8`, four tie-groups with
size multiset `{2, 3, 3, 4}`, `alpha = 1/3` in all three arms.

| arm | `(c, d)` per group | yield ratios | `pure_crit` | class |
|---|---|---|---:|---|
| `geom_v2_a3_pure0_homog` | `(0,2) (0,4) (2,1) (2,1)` | `2, 2` | 0 | homogeneous |
| `geom_v2_a3_pure0_hetero` | `(0,2) (0,3) (1,2) (3,1)` | `1/2, 3` | 0 | heterogeneous |
| `geom_v2_a3_pure_ge1` | `(0,3) (0,3) (2,0) (2,2)` | `inf, 1` | 1 | heterogeneous |

`alpha = 1/2` was requested but yields no non-degenerate triple at `H = 12`:
both candidates there have an arm with only one critical-bearing group, which
is null by construction rather than by the mechanism under test. `alpha = 1/3`
is used instead. This costs nothing, because `alpha` is demoted to a
descriptive variable under this protocol and `partial_feature_aliasing`
(`H = 12`, `alpha = 1/2`) is retained separately as the real-data anchor.

**Matching, stated honestly.** `pure0_homog` and `pure_ge1` are matched on `H`,
`A`, `alpha`, critical/distractor counts, group count, group-size multiset,
**and** per-group critical-count multiset `(2, 2)`. They differ in exactly one
respect: whether a critical-bearing group has zero distractors. This is the
clean test of H_pure.

`pure0_hetero` cannot match the per-group critical-count multiset -- it is
`(1, 3)` -- because at fixed group sizes, heterogeneous finite ratios *require*
unequal critical counts. This is a structural impossibility, not an oversight,
and it means the H_hetero discriminator is matched on every declared quantity
except that one. Its evidence is correspondingly weaker and must be reported as
such.

### Full scenario set

| scenario | role | `pure_crit` | ratio class |
|---|---|---:|---|
| `geom_v2_a3_pure_ge1` | A3 positive arm | 1 | heterogeneous |
| `geom_v2_a3_pure0_homog` | A3 null arm | 0 | homogeneous |
| `geom_v2_a3_pure0_hetero` | A3 discriminator | 0 | heterogeneous |
| `separable_shared_features` | positive anchor | 3 | homogeneous (`inf`) |
| `geom_holdout_a` | positive anchor | 1 | heterogeneous |
| `geom_holdout_b` | positive anchor | 1 | heterogeneous |
| `partial_feature_aliasing` | null anchor (M6) | 0 | homogeneous |
| `complete_feature_aliasing_negative_control` | invariant control | 0 | homogeneous |

The five non-A3 scenarios are copied verbatim from
`configs/geometry_v1.json` and may not be re-derived.

### Generation rule for the A3 arms

Deterministic, so the arms cannot drift:

1. Tie-groups are emitted in the order listed in the table above; positions are
   assigned to groups in ascending index order.
2. Features are one-hot indicators of group membership.
3. Within each group, critical positions precede distractor positions.
4. Critical-bearing groups are ranked by first position; the group of rank `j`
   assigns target action `j mod A` to **all** its critical positions. This
   guarantees within-group target consistency, hence realizability.
5. `minimum_matches = 2`; `iterations = 400`; `group_size = 48`; initial policy
   uniform.

**Mandatory realizability and learnability checks**, both at stage 2 and both
blocking: no tie-group may contain critical positions with conflicting target
actions, and the uniform baseline must achieve a success gain comparable to the
existing scenarios. `geom_dev_separable` failed the latter once already and had
to be redesigned; the check is not optional.

## Grids, hyperparameters, seeds

**Uniform `eta` grid** (18 points, floor `0.025`):

```
0.025, 0.03125, 0.0375, 0.05, 0.0625, 0.075, 0.10, 0.125, 0.15,
0.20, 0.25, 0.30, 0.40, 0.50, 0.625, 0.75, 1.00, 1.25
```

The floor is set by measurement, not convention: on the development sweep the
worst seed of `geom_dev_separable` required `eta = 0.0330` for the grid to
bracket RWP-OMD's distractor KL, and a floor of `0.05` dropped 19 of 40 seeds
there. `0.025` carries roughly 25% margin below the worst observed requirement.

**Methods**, all frozen, none re-selected:

| name | algorithm | `eta` | other |
|---|---|---:|---|
| `uniform_eta*` | `projected_group_omd` | grid | `projection_ridge = 0.0` |
| `rwp_eta125_lam3` | `rwp_omd` | 1.25 | `lambda = 3.0`, `mu = 0.0` |
| `projected_online_eta125` | `projected_online_rc_omd` | 1.25 | v1 reference |

Shared: `projection_steps = 60`, `projection_learning_rate = 0.5`,
`projection_tolerance = 1e-9`; reliability `decay = 0.9`,
`confidence_multiplier = 1.0`, `warmup_effective_samples = 8.0`,
`floor = 0.1`.

**Seeds.** Test seeds are `500-519` (20 seeds), disjoint from `geometry-v1`
test seeds `200-219`, from the `geometry_dev` selection seeds `0-19`, and from
the pre-freeze development seeds `400-439`. Twenty is justified by measurement:
on the development sweep the seed count needed to detect the observed effect at
80% power was 3, 3, and 4 for the three non-null scenarios. Per-seed values are
retained in full; aggregate means alone may not be reported.

## Metrics

### Primary -- frontier advantage

Per scenario, per seed `s`:

1. Let `D*(s)` be the candidate's cumulative distractor KL.
2. `i_lo` = the grid point with the **largest** distractor KL `<= D*(s)`.
3. `i_hi` = the grid point with the **smallest** distractor KL `>= D*(s)`.
4. Interpolate uniform AUC **linearly in log distractor KL**:

```
w        = (log D*(s) - log D_lo(s)) / (log D_hi(s) - log D_lo(s))
AUC_hat  = AUC_lo(s) + (AUC_hi(s) - AUC_lo(s)) * w
```

5. `Delta_frontier(s) = AUC_cand(s) - AUC_hat(s)`.

Brackets are defined by KL **rank**, not grid adjacency, so the rule stays well
defined where the per-seed `KL(eta)` curve is locally non-monotone and collapses
to the adjacent bracket wherever it is monotone.

**Degenerate cases.** If `D_hi(s) == D_lo(s)` the interpolation is skipped and
`AUC_lo(s)` is used. If `D_lo(s) <= 0` the same fallback applies.

### Secondary -- allocation efficiency

Identical machinery with critical KL as the matching axis and distractor KL as
the read-out, but interpolated **geometrically** (linear in log) in the
read-out as well:

```
log D_hat = log D_lo + (log D_hi - log D_lo) * w
ratio(s)  = D_cand(s) / D_hat(s)
```

Reported as a 95% CI on `log ratio`, exponentiated.

Geometric interpolation here is required, not stylistic: under complete
aliasing distractor KL equals critical KL identically, so matching on one and
reading the other must return the target itself. Linear-in-value interpolation
gave `0.9952` with a CI excluding `1.0` on the pre-freeze run -- a systematic
0.5% bias that the control correctly exposed. Geometric interpolation returns
exactly `1.00000`.

This metric guards the primary metric, which leaves critical KL unconstrained
and could otherwise be satisfied by spending more critical movement. Critical
KL is intended work; distractor KL is declared harm. The primary asks for more
benefit at equal harm; the secondary asks for less harm at equal intended work.

### Descriptive -- `dist/crit` KL ratio

Reported for mechanistic illustration only. **Excluded from all GO/NO-GO
decisions.** It is gameable: on the frozen v1 run the v1 algorithm scored
`-0.6385` against RWP-OMD's `-0.6673` while reducing distractor KL by nothing,
having inflated critical KL to `1.198x` baseline.

## Statistics

Paired per-seed differences, two-sided 95% t-confidence intervals, `df = n - 1`,
with t-criticals tabulated in the evaluator (no new runtime dependency).
Log-scale bounds for all ratio quantities.

**Dropped seeds.** A seed with `D*(s)` outside `[min_i D_i(s), max_i D_i(s)]` is
out of bracket, is dropped, and is counted. Dropping is non-random -- the
dropped seeds are those where the candidate suppressed distractor KL below
anything the grid reaches, i.e. its best seeds -- so a dropped-seed estimate is
biased downward and may not be reported as a point estimate. If more than **2 of
20** seeds are dropped in any scenario, that scenario's primary metric is
**INCONCLUSIVE (grid coverage failure)**, never a pass and never a fail.

## Decision rules

Per scenario, the primary verdict is:

- **positive** iff the 95% CI lower bound on `Delta_frontier` is `> 0`
- **negative** iff the upper bound is `< 0`
- **null** otherwise
- **inconclusive** if the dropped-seed limit is exceeded

The margin is zero rather than a positive constant. This is licensed by the
controls, not by convenience: on the development sweep the true effect at
`alpha = 0.8` was `+0.0038`, so any margin `>= 0.005` would reject a real,
reproducible effect, while the complete-aliasing control demonstrates the
zero-margin test does not fire spuriously.

### Pre-registered predictions

| scenario | H_pure predicts | H_hetero predicts |
|---|---|---|
| `geom_v2_a3_pure_ge1` | positive | positive |
| `geom_v2_a3_pure0_homog` | null | null |
| **`geom_v2_a3_pure0_hetero`** | **null** | **positive** |
| `separable_shared_features` | positive | positive |
| `geom_holdout_a`, `geom_holdout_b` | positive | positive |
| `partial_feature_aliasing` | null | null |
| `complete_feature_aliasing_negative_control` | null or negative | null or negative |

The single discriminating cell is `geom_v2_a3_pure0_hetero`.

### Outcomes

- **GO (H_pure confirmed)**: `pure_ge1` positive, `pure0_homog` null,
  `pure0_hetero` null, and every anchor matches its prediction.
- **REVISE (H_hetero confirmed)**: as above but `pure0_hetero` **positive**.
  This is a defined scientific outcome, not a failure. The published law
  becomes ratio heterogeneity, and `pure_crit` is recorded as its extreme case.
- **NO-GO (law refuted)**: `pure_ge1` null or negative, or any anchor
  contradicts both hypotheses.
- **HALT**: an invariant control fails (see below); no scientific conclusion is
  drawn and the implementation is treated as defective.
- **INCONCLUSIVE**: any scenario exceeds the dropped-seed limit.

Note that GO and REVISE are both informative and the protocol does not prefer
either. That is deliberate: the discriminating cell was chosen because it has
never been measured, and pre-committing to a preferred answer would defeat it.

## Negative controls

Two, of different kinds, and they are not interchangeable.

**Control 1 -- exact invariant (complete aliasing).** Under complete aliasing
critical and distractor KL are equal identically, so the secondary allocation
ratio must be `1.0` to within `1e-9`. This is a theorem about the
parameterization; violation is an implementation defect and forces **HALT**,
not a result.

**Control 2 -- no spurious pass (`pure_crit = 0` scenarios).** The primary
metric must not return positive on `geom_v2_a3_pure0_homog` or
`partial_feature_aliasing`. This is a calibration check on the zero margin.

The primary metric is **not** required to be zero under complete aliasing and
may be slightly negative: the pre-freeze run measured `-0.00286`, CI
`[-0.00351, -0.00222]`, plausibly because reliability-driven weight
fluctuation makes the effective step time-varying where there is no allocation
freedom to exploit. A small negative value is an accepted outcome. A positive
value is a Control 2 failure.

## What this protocol cannot establish

- **Nothing about non-linear function approximation.** Every result rests on
  assumption (M1). The collapse identity, the convexity argument, and the exact
  projection all depend on softmax-linearity. Extension is deferred.
- **Nothing about language-model RLVR.** No claim here transfers to a
  transformer without first defining and measuring a continuous analogue of
  `pure_crit`.
- **Nothing about why `partial_feature_aliasing` is null.** Lemma 2 gives a
  necessary condition; freedom (ii) above remains unexplained.
- **Nothing about `lambda` or `mu`.** Both stay frozen at the `geometry-v1`
  values; no selection occurs under this protocol.
- **The five inherited scenarios are seen data.** Only the three A3 arms are
  new.

## Archiving

- Raw output: `results/geometry_v2/` (gitignored).
- Golden records: `paper/frozen/geometry-v2-2026-08-18.json` and
  `paper/frozen/geometry-v2-2026-08-18-evaluation.json`.
- Each golden record carries: protocol ID, execution commit hash, SHA256 of the
  config file, SHA256 of the raw summary, seed list, and full per-seed arrays.
- `paper/CLAIMS.md` and `docs/ERRATA.md` are updated in a **separate** commit
  from any code or result commit.
