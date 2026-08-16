# Geometry V1 Protocol

Protocol ID: `geometry-v1-2026-08-15`

Status: **drafted, not frozen.** Freezing requires two commits, in order:

1. this document, the scenario configs, the evaluator, and its tests
   (structure frozen: scenarios, grids, statistics, selection rule, criteria);
2. the selected `lambda*` and `mu*` together with the full development-sweep
   record (values frozen).

Only after commit 2 may any test scenario be executed. Any result produced
before that is exploratory and must be labelled as such.

The algorithm under test is specified in
[`PARAMETER_SPACE_GEOMETRY_DESIGN.md`](PARAMETER_SPACE_GEOMETRY_DESIGN.md),
whose objective, gradient, weight normalization, and KL accounting are locked
and may not change under this protocol ID.

## Purpose

`function-approx-v1-2026-08-09` returned NO-GO: the Pareto behaviour of Online
RC-OMD held under separable shared features and failed under partial and
complete feature aliasing. The M6 mechanism diagnostic attributes the failure
to the projection, not the reliability estimator ([E3](ERRATA.md)).

This protocol asks one question:

> Does replacing the unweighted forward-KL projection with the
> reliability-weighted cross-entropy projection (RWP-OMD) recover the Pareto
> behaviour under **partial** aliasing, without regressing the separable case?

It does **not** ask whether complete aliasing can be fixed. Section
"Negative controls" explains why that is impossible by construction rather
than difficult.

## What this protocol cannot establish

Fixed in advance so it cannot be revisited afterwards.

- **The M6 scenarios are seen data.** Their outcome under the v1 algorithm is
  already known. They are an in-distribution diagnostic set. Only the two
  held-out scenarios carry generalization evidence.
- **RWP-OMD inherits nothing from `ood-v1` or `pareto-v1`.** Under one-hot
  features it performs arithmetic reliability interpolation where tabular
  Online RC-OMD performs geometric interpolation; the two differ by
  0.045--0.054 in policy at the frozen step 1.25
  ([design spec, Section 10](PARAMETER_SPACE_GEOMETRY_DESIGN.md)). Every claim
  must be re-earned here.
- **This is not a step-size frontier test.** `eta` is held fixed at the M6
  values precisely so that the comparison isolates the change of projection
  objective. A `pareto-v1`-style frontier version is deferred to a later
  protocol ID.
- **Nothing about language models.** GSM8K and every other LM experiment
  remain out of scope regardless of outcome.

## Scenario construction rule

A scenario is fully determined by
`(H, n_actions, critical_positions, target_actions, minimum_matches, tie_groups)`.

**Feature rule.** `tie_groups` partitions positions `0..H-1`. Features are the
one-hot indicator of tie-group membership, with groups ordered by their
smallest member index. This determines `F` uniquely, so the matrices below need
not be written out.

**Realizability constraint (mandatory).** Within any tie-group, all critical
members must share the same target action. Tied positions have identical
policies for every `theta`, so conflicting targets inside one group make the
optimum unrepresentable in the parametric family. This was not hypothetical: a
first draft of `geom_dev_separable` violated it and the baseline reached only
+0.10 success against +0.79 after correction, because its maximum achievable
match count (2) fell below its threshold (3). Every scenario below satisfies
`max_achievable_matches >= minimum_matches`, verified mechanically.

**Aliasing index.** `alpha = 1 - (1/H) * sum_g |c(g) - d(g)|`, defined in the
design spec, Section 14.

## Scenario sets

All three sets are disjoint. Shared settings: `group_size = 48`,
`iterations = 400`, `evaluation_interval = 5`, uniform initial policy.

### Development set (tuning only)

`H = 10`, `n_actions = 3`, `critical_positions = [0,1,4,6,9]`,
`target_actions = [2,2,1,1,1]`, `minimum_matches = 3`.

| Scenario | `alpha` | tie-groups |
|---|---:|---|
| `geom_dev_separable` | 0.000 | `[0,1] [2,3] [4,6,9] [5,7] [8]` |
| `geom_dev_partial_040` | 0.400 | `[0,2] [1] [3] [4,6] [5,7] [8,9]` |
| `geom_dev_partial_080` | 0.800 | `[0,2] [1,3] [4,5] [6,7,8] [9]` |
| `geom_dev_complete` | 1.000 | `[0,2] [1,3] [4,5] [6,7] [8,9]` |

### In-distribution diagnostic set (never used for tuning)

The three existing `configs/function_approx_preregistered.json` scenarios,
unchanged: `separable_shared_features` (`alpha = 0.000`),
`partial_feature_aliasing` (`alpha = 0.500`),
`complete_feature_aliasing_negative_control` (`alpha = 1.000`).

### Held-out set (never used for tuning)

| Scenario | `H` | `A` | `alpha` | critical | targets | `min_matches` | tie-groups |
|---|---:|---:|---:|---|---|---:|---|
| `geom_holdout_a` | 14 | 2 | 0.571 | `[1,3,6,8,11,13]` | `[1,0,1,0,1,1]` | 4 | `[0,1] [2,3] [4,6] [5,8] [7,9,10,12] [11,13]` |
| `geom_holdout_b` | 8 | 4 | 0.750 | `[0,3,5,6]` | `[3,1,2,0]` | 2 | `[0,1] [2,3] [4,5] [6] [7]` |

Horizons `{10, 14, 8}` and action counts `{3, 2, 4}` are all distinct from M6's
`(12, 2)`.

**Held-out validation disclosure.** Before freezing, every scenario above was
checked for learnability using the **projected uniform baseline only**, over 3
seeds, to confirm the task is well posed. Baseline success gains were +0.79
(all four dev), +0.66 (`geom_holdout_a`), +0.74 (`geom_holdout_b`). No
candidate-algorithm result informed any scenario's design, and no tuning
decision may cite these numbers.

## Methods

| Name | Algorithm | `eta` | Role |
|---|---|---:|---|
| `projected_uniform_eta075` | `ProjectedGroupOMD` | 0.75 | comparison baseline; M6's primary baseline |
| `projected_online_eta125` | `ProjectedOnlineReliabilityOMD` | 1.25 | v1 reference, the NO-GO algorithm; reported, never decisive |
| `rwp_omd_eta125` | `ReliabilityWeightedProjectionOMD` | 1.25 | candidate |

All paired comparisons are `rwp_omd_eta125` against `projected_uniform_eta075`,
matching M6's primary pair exactly so the two protocols are comparable.

**Frozen algorithm parameters** (unchanged from `ood-v1` / M6, may not move once
any result is observed): reliability decay 0.9, confidence multiplier 1.0,
warm-up effective samples 8, reliability floor 0.1, projection steps 60,
projection learning rate 0.5, projection tolerance 1e-9.

**Seeds.** Development `0..19`; test `200..219`. Disjoint blocks, 20 seeds per
cell throughout.

## Hyperparameter grid

| Parameter | Grid |
|---|---|
| `lambda` | `{1.0, 1.5, 2.0, 3.0, 5.0}` |
| `mu` | `{0.0, 1e-3, 1e-2}` |

15 combinations. Per the design spec Section 12: `lambda >= 1` only, `lambda = 1`
is the uniform-weight reference point, and the `mu` grid contains positive
values because `mu > 0` is what guarantees a unique minimizer.

The grid may not be extended after any development result is observed. If the
selected value sits on a grid boundary, that is reported explicitly and the
grid is still not extended under this protocol ID.

## Statistics

Adopted unchanged from `pareto-v1-2026-08-14`, which replaced M6's point
estimates with paired confidence bounds.

For each scenario and seed `s`, against the baseline:

```
d_AUC(s) = AUC_rwp(s) - AUC_uniform(s)
r_KL(s)  = distractorKL_rwp(s) / distractorKL_uniform(s)
```

- **AUC non-inferiority.** Margin `delta = 0.02`, matching `function-approx-v1`
  because this is the same task family. Holds when the lower limit of the
  two-sided 95% t-CI on `mean(d_AUC)` exceeds `-delta`, `df = 19`.
- **Distractor KL bound.** `rho = 0.75`, matching `function-approx-v1`. `r_KL`
  is right-skewed, so the test is on the log scale: holds when
  `exp(upper limit of the 95% t-CI on mean(log r_KL)) <= rho`.
- **Degenerate seeds.** If `distractorKL_uniform(s) < 1e-9`, `r_KL(s)` is
  undefined and that seed is dropped from the KL test only, with the count
  reported. More than 2 of 20 dropped makes the cell's KL test
  **inconclusive**, which counts as a failure.
- **Power failure is not a pass.** If the 95% CI half-width on `d_AUC` exceeds
  `delta` in a scenario, that scenario is **inconclusive**, not a pass. The
  protocol result is then incomplete and the seed count must be raised under a
  new protocol ID.

A scenario **passes** when the AUC and KL conditions both hold.

## Selection rule (development set only)

Applied mechanically to the 15 grid points, using development scenarios only.

`geom_dev_complete` is **excluded from selection**: at `alpha = 1` critical and
distractor KL are provably equal for every candidate (design spec, Section 14),
so it carries no discriminating signal. It is used solely to assert the
invariant.

1. **Regression gate.** Discard any grid point failing `geom_dev_separable`.
   A candidate that breaks the case v1 already handles is not eligible.
2. **Primary.** Among survivors, maximize the number of passing scenarios
   among `{geom_dev_separable, geom_dev_partial_040, geom_dev_partial_080}`.
3. **Tie-break 1.** Minimize the mean distractor-KL upper bound across those
   three scenarios.
4. **Tie-break 2.** Smallest `lambda`.
5. **Tie-break 3.** Smallest `mu`.
6. **Tie-break 4.** First in grid order, `lambda` major and `mu` minor.

Steps 4--6 guarantee a unique deterministic winner. If **no** grid point
survives step 1, the protocol terminates at NO-GO without any test scenario
being run, and that outcome is reported.

## Pre-declared decision rule

Evaluated on the diagnostic and held-out sets after `lambda*`, `mu*` are frozen.

**GO** requires all four of:

1. `partial_feature_aliasing` (M6, `alpha = 0.500`) **passes** -- the specific
   failure this work targets;
2. `geom_holdout_a` **and** `geom_holdout_b` both **pass** -- generalization;
3. `separable_shared_features` (M6, `alpha = 0.000`) **passes** -- no
   regression on the case v1 already handled;
4. the complete-aliasing invariant holds (below).

Any scenario **inconclusive** makes the overall result **inconclusive**, not a
pass. Anything else is **NO-GO**.

This is deliberately strict. The claim being attempted is positive, made after
a recorded NO-GO, so a majority-vote rule would be too weak to support it.

## Negative controls

1. **Complete aliasing (`alpha = 1`).** At `alpha = 1` every tie-group has
   `c(g) = d(g)`, so realized critical KL equals realized distractor KL exactly,
   for any update in this parametric family (proof in design spec, Section 14).
   Applies to `complete_feature_aliasing_negative_control` and
   `geom_dev_complete`.
   - Assertion: `|criticalKL - distractorKL| <= 1e-9` per seed.
   - This is **not** a success criterion. Equal KL is the expected correct
     outcome. An apparent distractor-KL *improvement* here is evidence of a
     bug and must halt the protocol pending investigation.
2. **Zero reliability.** `r = 0` everywhere must produce zero policy movement
   (design spec, Section 9). Enforced as a unit test, not a scenario.
3. **Inverted lambda.** A single `lambda = 0.5` point is run on the development
   set as a labelled negative check. `lambda < 1` gives *less* weight to less
   reliable positions, so it should not outperform `lambda = 1`. It is never a
   selection candidate; if it wins on the primary metric, the implementation or
   the objective is suspect.

## Required deliverables before freezing

- `configs/geometry_dev.json` -- development sweep, 15 grid points plus the
  `lambda = 0.5` negative check and both baselines.
- `configs/geometry_v1.json` -- frozen test config, written only after
  `lambda*`, `mu*` are selected.
- `algorithms/geometry_omd.py` -- `ReliabilityWeightedProjectionOMD`.
- `experiments/evaluate_geometry_protocol.py` -- implements the statistics,
  selection rule, and decision rule above in code, before any result exists.
- `tests/test_geometry_omd.py` -- the ten invariants in design spec Section 13.
- `tests/test_evaluate_geometry_protocol.py` -- decision-rule tests.

## Execution discipline

1. Commit this protocol, the scenario configs, the algorithm, the evaluator,
   and all tests. **Until this happens nothing is frozen.**
2. Run the development sweep once. Apply the selection rule mechanically.
3. Commit `lambda*`, `mu*`, and the complete development-sweep record,
   including every grid point and both negative checks.
4. Run the test config once. No parameter changes.
5. Preserve per-seed histories under `results/geometry_v1/`.
6. Archive the summary and protocol evaluation under `paper/frozen/`, with
   protocol ID, execution commit, config hash, Python and NumPy versions, and
   SHA256 of every generated file. `results/` is gitignored, so the per-point
   evaluation must be archived under version control.
7. Report every scenario and every grid point, including failures.
8. Any later change to scenarios, grids, margin, bound, selection rule, or
   decision rule requires a new protocol ID and is post-hoc.

## Interpretation fixed in advance

A **GO** supports exactly this statement:

> Replacing the unweighted forward-KL projection with the reliability-weighted
> cross-entropy projection recovers Pareto behaviour under partial feature
> aliasing on the M6 partial scenario and on two held-out shared-feature
> scenarios, at fixed step sizes, without regressing the separable case.

A **GO** does **not** support: any claim at complete aliasing; reward
improvement at equal nominal step size; a step-size frontier claim;
generalization beyond shared linear features; causal credit identification; or
any statement about language-model RLVR.

A **NO-GO** means the reliability-weighted projection objective does not by
itself resolve the M6 bottleneck, and the parameter-space direction must be
reconsidered rather than scaled up.
