# Claims and evidence boundary

Corrections to previously published claims are recorded in
[`docs/ERRATA.md`](../docs/ERRATA.md). Read both files together.

The four categories below are deliberately distinct. In particular, "tested and
not supported" is not the same as "not yet tested", and conflating them
overstates how much of the hypothesis space remains open.

## Supported by the frozen OOD protocol

Protocol `ood-v1-2026-08-08`, execution commit `d37a5ff`, 4 scenarios, 10 seeds.

**Reproduction status: confirmed.** Regenerated at the execution commit on
2026-08-14; all deterministic metrics match the published table to within 5e-7,
and the GO decision is unchanged. Golden record:
`paper/frozen/ood-v1-2026-08-08.json`.

- Online RC-OMD passed all four pre-declared controlled scenarios.
- Its normalized success AUC was within 0.0018 of the selected Uniform Group OMD
  comparison in every scenario.
- It used 18.9%--24.8% as much absolute cumulative distractor-position KL.
- Its CPU runtime ratio was comfortably below the frozen 1.5 feasibility
  threshold in every scenario. **No specific runtime interval may be quoted**:
  wall-clock ratios do not reproduce across machines, and one scenario inverted
  on re-run. See [E7](../docs/ERRATA.md).

The primary pair uses different base steps for the two methods, selected on
development tasks. This is a Pareto-matched comparison, not a same-step
comparison.

## Supported by the frozen Pareto V1 protocol

Protocol `pareto-v1-2026-08-14`, execution commit `751c3f6`, 4 scenarios, 20
seeds, 7-point step grid (`{0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00}`).
Original execution; there is no earlier published table to reproduce against.
Golden record: `paper/frozen/pareto-v1-2026-08-14.json`; full per-point
breakdown in `paper/frozen/pareto-v1-2026-08-14-evaluation.json`.

This protocol exists because the OOD result above used one hand-picked step
pair per method, selected on development data -- see
`docs/PARETO_V1_PROTOCOL.md` for the loophole this closes.

- **GO.** All 4 scenarios passed; every one of the 28 Uniform grid points
  tested (7 steps x 4 scenarios) was covered by the deterministic
  smallest-qualifying-step matching rule.
- At every covered point, Online RC-OMD's success AUC was within the 0.01
  non-inferiority margin of the matched Uniform step (paired 95% t-CI lower
  bound, `df=19`), and its distractor KL was at most 75% of the matched
  Uniform step's, with a 95% upper confidence bound on the log-scale ratio.
- The matched Online step never sat at the grid maximum (2.00) in any
  scenario, so the frontier claim is not truncated by the grid's edge.
- Systems feasibility passed: median matched-pair runtime ratio was well
  under the frozen 1.5 threshold in every scenario. **No specific runtime
  interval may be quoted**, for the same reason as [E7](../docs/ERRATA.md).

This upgrades the OOD claim above from "holds at one selected step pair" to
"holds across a pre-declared step-size frontier, with step size as a swept
nuisance parameter rather than a selected one." It is **not** a new
generalization test: it reuses the four `ood-v1` scenarios and says nothing
about task families outside them, nor about reward improvement at equal
nominal step size ([E1](../docs/ERRATA.md)), nor about shared-parameter
function approximation, which remains NO-GO below.

## Supported by the frozen geometry-v1 protocol

Protocol `geometry-v1-2026-08-15`, execution commit `63145da`, 5 scenarios,
20 seeds, `lambda*=3.0` and `mu*=0.0` selected on separate development
scenarios and frozen before any test scenario ran. Golden record:
`paper/frozen/geometry-v1-2026-08-15.json`; per-scenario breakdown in
`paper/frozen/geometry-v1-2026-08-15-evaluation.json`.

The algorithm is **RWP-OMD**, a reliability-weighted forward cross-entropy
projection. It is *not* Online RC-OMD and inherits nothing from `ood-v1` or
`pareto-v1`; see `docs/PARAMETER_SPACE_GEOMETRY_DESIGN.md` Section 10.

- **GO.** All 4 required scenarios passed and the complete-aliasing invariant
  held exactly.
- `partial_feature_aliasing`, the scenario that produced the M6 NO-GO, passed
  with a distractor-KL upper bound of **0.597** against the 0.75 bound. The v1
  algorithm measured **1.030** on the same run.
- Both held-out scenarios passed (**0.433**, **0.397**); v1 measured 0.841 and
  0.790 on the same run, i.e. would have failed.
- `separable_shared_features` did not regress (0.080).
- **Mechanism, as measured against the fixed baseline step.** Under partial
  aliasing RWP-OMD cut distractor KL to 0.575 of baseline while cutting
  critical KL only to 0.709. v1 did the reverse, raising critical KL to 1.198
  while leaving distractor KL at 1.000. **This may not be read as "not merely
  smaller steps"** for that scenario: against a step-swept uniform frontier the
  effect is null, and RWP-OMD is indistinguishable from v1 on allocation there.
  See [E9](../docs/ERRATA.md). The selective-protection reading does survive on
  the two held-out scenarios.

Limits, all fixed before execution:

- **`eta` was held fixed** (candidate 1.25, baseline 0.75). This does **not**
  establish that RWP-OMD beats a step-size-matched projected uniform baseline.
  No step frontier was swept. This is the single largest open confound and is
  the deferred `geometry-v2` question. The result **may not be described as
  step-robust**.
- **AUC is uniformly below baseline**, by 0.0114 to 0.0176 against a 0.020
  non-inferiority margin. This is a Pareto trade-off with little headroom, not
  a reward improvement.
- The three M6 scenarios are **seen data**; only the two held-out scenarios
  carry generalization evidence.
- Complete aliasing is a negative control and contributed nothing to the
  decision.

## Tested under the frozen geometry-v2 protocol

Protocol `geometry-v2-2026-08-18`, execution commit `c5384a5`, 8 scenarios, 20
seeds, 18-point step grid (`0.025` to `1.25`). Golden record:
`paper/frozen/geometry-v2-2026-08-18.json`.

This protocol swept the step size that `geometry-v1` held fixed, closing the
confound named as that protocol's largest limitation. Its primary metric is the
AUC advantage over the uniform frontier interpolated at RWP-OMD's own distractor
KL.

- **Frontier advantage exists and is not a step-size artifact.** RWP-OMD lies
  above the step-swept uniform frontier on 6 of 8 scenarios, by +0.0036 to
  +0.1228 after bias correction. This is the claim `geometry-v1` could not make.
- **NO-GO on the mechanism.** The protocol's pre-registered hypothesis -- that a
  pure-critical tie-group is what produces the advantage -- is **refuted**. In an
  exactly-matched triple (`H=12`, `alpha=1/3`, matched on group sizes, critical
  counts, and every declared quantity), the arm with `pure_crit = 0` and
  homogeneous yield ratios still showed +0.00494 [+0.00418, +0.00570], where both
  pre-registered hypotheses predicted null.
- **The advantage is graded, not binary.** Bias-corrected, inside the matched
  triple: `pure_ge1` +0.01209, `pure0_hetero` +0.00594, `pure0_homog` +0.00398.
  Pure-criticality and ratio heterogeneity each contribute; neither is necessary.
- **`partial_feature_aliasing` remains null**, consistent with [E9](../docs/ERRATA.md).
- **Complete aliasing** returned -0.00297 on the primary metric and **exactly
  1.000000** on the secondary allocation invariant.

Limits:

- The primary metric has a measured positive bias of +0.00136 from interpolating
  a concave curve linearly ([E10](../docs/ERRATA.md)). Effects below roughly
  +0.002 are not resolvable. The NO-GO survives on both raw and corrected values.
- **No replacement law is asserted.** The graded pattern above is post-hoc on data
  now spent for hypothesis testing. Any successor hypothesis must be tested on
  new scenarios.
- Softmax-linear one-hot tie-group features only. Nothing transfers to non-linear
  function approximation or to language-model RLVR.
- `lambda*` and `mu*` were inherited frozen and not re-selected; this says nothing
  about their optimality.

## Supported by the frozen geometry-v3 protocol

Protocol `geometry-v3-2026-08-19`, 50-scenario randomized family plus a
complete-aliasing control, 20 seeds, split 30 discovery / 20 confirmation by a
committed seed **before any scenario ran**. Golden records:
`paper/frozen/geometry-v3-2026-08-19-discovery.json` and
`-confirmation.json`.

- **CONFIRMED: `alpha` predicts the frontier advantage.** Spearman
  `rho = -0.8859`, 95% bootstrap interval `[-0.9519, -0.7210]`, on 20 held-out
  scenarios never read during selection. Discovery measured `-0.7868` on the
  disjoint 30. Higher aliasing, smaller advantage.
- **The selection was not overfitted.** `pure_crit` and `pure_crit_indicator`,
  refuted by `geometry-v2` and retained in the closed menu precisely as a trap,
  ranked fifth and sixth (0.4785, 0.4621) and were not selected. The held-out
  correlation is stronger than the discovery correlation.
- Both controls held: complete-aliasing allocation invariant exact at 2.22e-16,
  pooled leave-one-out bias +0.00142 (positive, as [E10](../docs/ERRATA.md)
  requires).

Limits:

- **The confirmation criterion was amended after the confirmation subset ran**
  ([E11](../docs/ERRATA.md)). The frozen wording was self-contradictory for
  negative predictors. The correction is the only self-consistent reading and
  the threshold did not move, but it is post hoc and may be discounted.
- **Correlation, not mechanism.** No causal account is claimed.
- **The discovery margin over `critical_mass_in_pure_groups` was 0.0024**, a tie
  in all but the letter of the deterministic rule. This licenses `alpha` alone
  and does not establish which of the two is mechanistically prior.
- **Assumption (M1) only** -- softmax-linear one-hot tie-group features. Whether
  `alpha` predicts anything once linearity is dropped is the open Stage B
  question.
- This reverses `geometry-v2`'s demotion of `alpha` to a descriptive variable,
  which was itself an inference from nine scenarios.

## Supported only by exploratory development experiments

- Entropy can hurt when its alignment with positional importance is reversed
  (M2 entropy-misleading, AUC gap -0.0148 under common random numbers).
- Entropy can help when aligned (M2 entropy-aligned, gap +0.0029 under common
  random numbers). The direction survives paired re-sampling, but the magnitude
  is small and no interval has been computed.
- Online moments are much cheaper than the implemented per-group bootstrap
  estimator.
- Reliability decay changes a ranking-lag trade-off.

Both entropy claims were re-run after the M2 seeding defect was fixed and both
kept their sign; see [E5](../docs/ERRATA.md) and
`docs/MILESTONE_2_CORRECTED_REANALYSIS.md`.

## Tested and not supported

These were measured. They are not open questions.

- **Improved reward at a fixed nominal step size.** Uniform Group OMD had higher
  AUC in all 12 archived same-step cells: 8 in the M4 matched sweep, 4 in the M5
  same-step diagnostic. In the same cells Online RC-OMD used substantially less
  absolute distractor KL, so the finding is a Pareto trade-off, not a plain
  regression. See [E1](../docs/ERRATA.md).
- **Benefits under shared-parameter function approximation.** Protocol
  `function-approx-v1-2026-08-09` returned **NO-GO** at 1/3 scenarios. The
  Pareto behavior held under separable features and did not hold under partial or
  complete feature aliasing. All AUC conditions passed; the failure was entirely
  in the declared distractor-KL condition. Reproduction confirmed at execution
  commit `2c91c69`; golden record
  `paper/frozen/function-approx-v1-2026-08-09.json`.
  **This NO-GO stands unchanged.** It is a statement about Projected Online
  RC-OMD, the v1 algorithm. A different algorithm (RWP-OMD) later passed on
  these scenarios under `geometry-v1`, below; that does not retroactively
  change what v1 did, and the two must not be conflated.

Neither result refutes the general statement. Both bound the region where the
current algorithm is known not to deliver.

## Mechanism findings that survive the negative results

- Under feature aliasing the reliability estimator degraded only mildly:
  critical/distractor separation 2.93 to 2.59, top-k precision 0.725 to 0.704.
  The M6 failure is attributable to the parametric projection's inability to
  realize independent local trust regions, not to estimator collapse. See
  [E3](../docs/ERRATA.md).

## Not yet tested

- Causal identification of step credit.
- Robustness under persistent confounding or nonstationarity.
- Benefits for language-model RLVR.
- ~~Whether RWP-OMD's advantage survives step-size matching.~~ **Tested**; it
  does, on 6 of 8 scenarios. See the geometry-v2 section above.
- ~~**What structural property predicts the frontier advantage.**~~ **Tested**;
  `alpha` predicts it under assumption (M1). See the geometry-v3 section. What
  *mechanism* produces it remains unknown -- geometry-v3 establishes a correlate,
  not a cause.
- Reliability geometry beyond shared *linear* features.

## Known defects affecting published numbers

Recorded in [`docs/ERRATA.md`](../docs/ERRATA.md).

- Bootstrap uncertainty was measured partly outside the update-relevant
  action-difference subspace, making M3 reliability more conservative than the
  update geometry justifies ([E4](../docs/ERRATA.md)). **Status: fixed and
  measured.** The paired re-run is complete; for local RC-OMD the correction is
  a Pareto improvement (every variant gained AUC and reduced distractor KL). See
  `docs/MILESTONE_3_CORRECTED_REANALYSIS.md`. The M3 headline conclusion is
  unchanged: RC-OMD v1 still does not outperform Uniform Group OMD overall.
- M2 seeded trajectories per method, so its comparisons were unpaired and
  depended on config method order ([E5](../docs/ERRATA.md)). M3--M6 are
  unaffected. **Status: fixed and measured**; the M2 gap figures quoted above
  (`+0.0029`, `-0.0148`) are already the corrected, paired values. See
  `docs/MILESTONE_2_CORRECTED_REANALYSIS.md`.
- M3's "81%--83% distractor reduction" is a reduction in KL *fraction*, not
  absolute KL ([E2](../docs/ERRATA.md)). This is a labelling distinction, not a
  re-run: absolute values (89.9%--92.0%, post-E4) are recorded in
  `docs/MILESTONE_3_CORRECTED_REANALYSIS.md`.

Any abstract, presentation, or submission must preserve these distinctions.
