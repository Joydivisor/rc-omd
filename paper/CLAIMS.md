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

## Supported only by exploratory development experiments

- Entropy can hurt when its alignment with positional importance is reversed
  (M2 entropy-misleading, AUC gap -0.0157).
- Online moments are much cheaper than the implemented per-group bootstrap
  estimator.
- Reliability decay changes a ranking-lag trade-off.

Provisional within this category, per [E5](../docs/ERRATA.md):

- That entropy *helps* when aligned. The measured gap is 0.0028 AUC, and M2 used
  method-dependent seeding, so the comparison was not paired. Pending a
  common-random-number re-run.

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
- Reliability expressed in parameter space rather than as per-position scalars.

## Known defects affecting published numbers

Recorded in [`docs/ERRATA.md`](../docs/ERRATA.md); none are corrected in the
numbers above, because the corrected runs have not been executed.

- Bootstrap uncertainty is measured partly outside the update-relevant
  action-difference subspace, making M3 reliability more conservative than the
  update geometry justifies ([E4](../docs/ERRATA.md)).
- M2 seeds trajectories per method, so its comparisons are unpaired and depend on
  config method order ([E5](../docs/ERRATA.md)). M3--M6 are unaffected.
- M3's "81%--83% distractor reduction" is a reduction in KL *fraction*, not
  absolute KL ([E2](../docs/ERRATA.md)).

Any abstract, presentation, or submission must preserve these distinctions.
