# Geometry V1 Protocol

Protocol ID: `geometry-v1-2026-08-15`

Status: **reviewed; document frozen as of this commit.** The scenarios, grids,
statistics, selection rule, decision rule, and failure semantics below are
closed. They may not be altered under this protocol ID; a change requires a new
one and is post-hoc with respect to anything already observed.

Freezing completes in three stages, of which the first is done:

1. **Document frozen (this commit).** Review items resolved: strict GO rule
   retained; `eta` scope fixed with `geometry-v2` named as the conditional
   frontier successor; power clause rewritten and given both an AUC and a KL
   precision floor; scenario counts disambiguated; per-condition failure
   semantics defined.
2. **Structure frozen.** Scenario configs, the algorithm, the evaluator, and
   its tests are committed, implementing this document verbatim. Those
   artifacts may not introduce or relax any rule stated here; if implementation
   shows a rule to be wrong, it must be amended in an explicit commit *before*
   any sweep is run, never silently in code.
3. **Values frozen.** The selected `lambda*` and `mu*` are committed together
   with the complete development-sweep record.

Only after stage 3 may any test scenario be executed. Any result produced
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
  objective. **Scope decision, confirmed before freezing:** the frontier
  version is a named successor, `geometry-v2`, modelled on
  `pareto-v1-2026-08-14`, and it is **conditional on `geometry-v1` returning
  GO**. Running a step-size frontier over an objective that does not work at
  its own operating point would measure nothing, so a `geometry-v1` NO-GO
  sends the work back to design rather than on to `geometry-v2`. Consequently
  every `geometry-v1` claim is conditional on the selected step pair, exactly
  the limitation `pareto-v1` was created to remove from `ood-v1`, and it may
  not be described as step-robust.
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

### Scenario count and what was validated

The protocol involves **nine** scenarios in total: **six new** ones introduced
here (four development, two held-out) and **three pre-existing** M6 diagnostic
scenarios carried over unchanged from `configs/function_approx_preregistered.json`.

Only the **six new** scenarios were designed and validated in this work. The
three M6 scenarios are reused verbatim and were not re-validated: they were
already executed under `function-approx-v1-2026-08-09`, and modifying or
re-screening them would break comparability with that recorded NO-GO.

**Validation disclosure (six new scenarios only).** Before freezing, each new
scenario was checked for (a) realizability, mechanically, and (b) learnability,
using the **projected uniform baseline only**, over 3 seeds, to confirm the task
is well posed. Baseline success gains were +0.79 for each of the four
development scenarios, +0.66 for `geom_holdout_a`, and +0.74 for
`geom_holdout_b`. No candidate-algorithm result informed any scenario's design,
and no tuning decision may cite these numbers.

Of the nine, **five** are executed in the test phase (three M6 diagnostic plus
two held-out); the four development scenarios appear only in the tuning sweep.

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
  **inconclusive**, and the scenario is then **inconclusive** rather than
  failed.

  > **Amendment, freeze stage 2.** As first written this clause ended "which
  > counts as a failure", inherited from `pareto-v1` where an inconclusive KL
  > test had to resolve to something for coverage counting. That is
  > inconsistent with the power clause below, which was rewritten in the same
  > document to route imprecise *failures* to `inconclusive`. Discarding more
  > than a tenth of the sample and then reporting "the algorithm failed" is
  > precisely the misreading the power clause exists to prevent, so the two
  > are reconciled in favour of `inconclusive`. Recorded here rather than
  > resolved silently in the evaluator, per the stage-2 rule that a defective
  > stated rule must be amended in its own commit before any sweep is run. No
  > sweep had been run when this was written.
A scenario **passes** when the AUC and KL conditions both hold.

### Power clause

`pareto-v1` phrased its power clause as preventing a *pass* produced by
confidence intervals too wide to exclude anything. That rationale does not
transfer to this protocol and is not reused. Both conditions here are
**conservative under widening**: a wider CI lowers the AUC lower limit and
raises the KL upper limit, so widening can only ever make a scenario harder to
pass. A pass is therefore a pass regardless of CI width, and no precision
requirement is imposed on passing scenarios.

The real hazard is the opposite one: reading a *failure* caused by insufficient
precision as evidence that the effect is absent. The clause therefore applies
**only to failing scenarios**. A failing scenario is declared **inconclusive**
rather than failed when it was not measured precisely enough to have detected
the effect under test:

| Condition that failed | Declared inconclusive when |
|---|---|
| AUC non-inferiority | 95% CI half-width on `d_AUC` exceeds `delta` (0.02) |
| Distractor KL bound | 95% CI half-width on `mean(log r_KL)` exceeds `log 2` (0.693), i.e. the ratio is uncertain by more than a factor of two |

Both thresholds are fixed here and may not be adjusted after any result is
observed. An inconclusive scenario is neither a pass nor a failure; it makes
the overall protocol result inconclusive and requires a higher seed count under
a new protocol ID.

**This clause is expected to be inactive.** Measured on the M6 scenarios at
`n = 10` and projected to the protocol's `n = 20`, achievable half-widths are
0.00035--0.00039 on `d_AUC` against `delta = 0.02`, and 0.017--0.100 on
`log r_KL` against `log 2`. That is roughly 50x margin on the AUC condition and
7--40x on the KL condition. These tasks are measured far more precisely than the
margins require, so the clause is a safety net rather than an operating
constraint, and any failure it does not catch should be read as a real failure.
The measurement used only the baseline and the already-published v1 algorithm;
no candidate result informed it.

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

**Decision confirmed before freezing: the strict conjunction is retained.** A
relaxation to 3-of-4 was considered and rejected. The claim being attempted is
positive and follows a recorded NO-GO, so a majority-vote rule would be too
weak to support it; and the usual objection to strict conjunctions -- that
accumulated noise across four tests produces spurious failures -- does not
apply at the precision measured here (roughly 50x margin on the AUC condition,
7--40x on the KL condition; see the power clause). The binding quantity is the
KL point estimate, not sampling noise: M6-partial currently sits at a
distractor-KL ratio near 0.97 and must reach 0.75, a reduction of roughly a
quarter that no plausible noise realization will supply by accident.

### What each failure means

Fixed in advance so that a NO-GO cannot be reinterpreted after the fact. Every
row is a NO-GO unless stated otherwise; the rows differ in what should happen
next.

| Failing condition | Reading | Required next action |
|---|---|---|
| **M6 partial fails**, both held-out pass | The objective does not fix the specific failure it was designed for, yet generalizes elsewhere. Treat as suspicious rather than encouraging: the M6 partial structure (`alpha = 0.500`, critical-dominant mixed groups) has some property the held-out structures lack. | Characterize that structural difference before any further algorithm work. Do not scale up. |
| **One or both held-out fail**, M6 partial passes | The fix is specific to seen data and does not generalize. Most likely `lambda*` is overfit to the development set despite the disjointness discipline. | Reconsider the parameter-space formulation. A wider development set alone is not a sufficient response. |
| **Separable regresses** | The objective damages the case v1 already handled, meaning reliability weighting is harmful where there is no conflict to arbitrate. Most likely sources: the mean-normalized weights, or the arithmetic-vs-geometric target change. | Diagnose and fix before any other result is interpreted. A GO on other scenarios does not offset this. |
| **Complete-aliasing invariant violated** | Not a research result. `alpha = 1` forces equal critical and distractor realized KL as a theorem (design spec, Section 14), so a violation is an implementation defect. | Halt the protocol. Do not report any other scenario until resolved. |
| **No development grid point survives the regression gate** | The objective cannot preserve the separable case at any `(lambda, mu)` in the grid. The formulation is wrong, not merely untuned. | Terminate at NO-GO **without running any test scenario**, and return to the design specification. |
| **Any scenario inconclusive** | Insufficient precision, not evidence of absence. This is **not** a NO-GO. | Raise the seed count under a new protocol ID. The existing result may not be quoted as a negative finding. |

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

1. Commit the scenario configs, the algorithm, the evaluator, and all tests,
   implementing this document verbatim (freeze stage 2; this document itself
   was frozen at stage 1). **Until this happens the structure is not frozen
   and no sweep may be run.**
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
