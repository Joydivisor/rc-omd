# Pareto Frontier Protocol

Protocol ID: `pareto-v1-2026-08-14`

Status: **frozen.** This commit adds this protocol document together with
`configs/pareto_v1.json`, `experiments/evaluate_pareto_protocol.py`, and the
pairing assertion tests (`tests/test_pareto_pairing.py`), satisfying the
pre-registration requirement below. Execution must happen at or after this
commit; any result produced before it is exploratory and must be labelled as
such.

## Purpose

`ood-v1-2026-08-08` compared two hand-picked points: Online RC-OMD at base step
1.25 against Uniform Group OMD at base step 0.75. Both were selected from the
Milestone 4 development sweep after that sweep was observed. The GO result is
therefore conditional on a step-size pair chosen with knowledge of the
development data.

This protocol replaces the point comparison with a frontier comparison, and fixes
the specific loophole that a frontier comparison introduces: if the analyst may
search the frontier for a favourable pairing, the frontier becomes a larger
post-hoc selection space than the single pair it replaced.

The mechanism used here is a **deterministic matching rule**. Exactly one Online
step is compared against each Uniform step, chosen by a rule fixed in advance.

## What this protocol does and does not test

It tests whether the Pareto claim survives when step size is treated as a swept
nuisance parameter rather than a selected one.

It is **not** a new generalization test. It runs on the four `ood-v1` scenarios,
which have already been used. A pass upgrades nothing about generalization to
unseen task families; it only removes the step-selection confound from a claim
already made on those tasks.

It tests the existing `OnlineReliabilityOMD` only. Mean-one normalized
reliability is a different algorithm and must not enter this protocol.

## Frozen algorithm parameters

Unchanged from `ood-v1-2026-08-08`:

- reliability decay: 0.9
- confidence multiplier: 1.0
- warm-up effective samples: 8
- reliability floor: 0.1

No value above may change after any result is observed.

## Frozen design

- **Scenarios:** the four `ood-v1` scenarios, unchanged.
- **Step grid:** `{0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00}`, identical for both
  methods. Spacing 0.25. Seven points per method per scenario.
- **Seeds:** 20 per cell, `seed = 0..19`.
- **Pairing:** the same seed must produce the same environment stochasticity for
  every method and every step size. `experiments/run_reliability_diagnostics.py`
  already seeds with `np.random.default_rng(seed)` and does not mix the method
  index into the seed, so this holds. It must be asserted in a test, not assumed.

Seeds are raised from 10 to 20 because the effects under test (AUC differences
around 0.005) are small relative to the non-inferiority margin (0.01), and
because a frontier test makes many more comparisons than a point test.

## Metrics

Per cell (scenario x method x step x seed):

- normalized exact-success AUC
- cumulative absolute KL on known distractor positions
- cumulative KL on pivotal positions
- harmful-update rate
- CPU runtime

Absolute distractor KL is primary. The KL *fraction* may be reported but must
never be the basis of a decision, per [E2](ERRATA.md).

## Statistical rules

All rules below are fixed before execution.

### Paired differences

For a Uniform step `u` and an Online step `o` in the same scenario, and for each
seed `s`:

```text
d_AUC(s)  = AUC_online(o, s) - AUC_uniform(u, s)
r_KL(s)   = KL_online(o, s) / KL_uniform(u, s)
```

Both are computed within seed. Unpaired comparisons of group means are not used
anywhere in this protocol.

### AUC non-inferiority

Margin `delta = 0.01`, matching `ood-v1`.

Non-inferiority holds when the **lower limit of the two-sided 95% t-confidence
interval on the mean of `d_AUC`** is greater than `-delta`, with `df = 19`.

This is a non-inferiority test, not a comparison of point estimates. A mean
difference above `-delta` with a confidence interval extending below `-delta`
does **not** pass.

### Distractor KL bound

Bound `rho = 0.75`, matching `function-approx-v1`.

`r_KL` is positive and right-skewed, so the test is applied on the log scale. The
KL condition holds when:

```text
exp( upper limit of the two-sided 95% t-CI on mean( log r_KL ) ) <= rho
```

This is an upper confidence bound on the ratio, not the ratio of the means.

Degenerate seeds: if `KL_uniform(u, s) < 1e-9` for any seed, `r_KL(s)` is
undefined for that seed. Such seeds are dropped from the KL test only, their
count is reported per cell, and if more than 2 of 20 seeds are dropped the cell's
KL test is declared **inconclusive**, which counts as a failure.

### Deterministic matching rule

For each Uniform step `u`, the matched Online step is:

> the **smallest** step in the grid at which AUC non-inferiority against `u`
> holds.

Only that one Online step is tested against `u` for the KL condition. If several
Online steps would satisfy the KL condition, this is irrelevant: the rule already
selected one, before the KL numbers were consulted.

The rule is deliberately AUC-first and minimum-step. Choosing the smallest
qualifying step is the conservative direction for the KL test, because larger
steps produce more KL.

### Coverage and the anti-shrinkage rule

A Uniform grid point `u` is **covered** when its matched Online step exists and
the KL condition holds at that step.

If no Online step in the grid achieves AUC non-inferiority against `u`, then `u`
is **uncovered** and counts as a **failure**. It is not excluded, and the grid is
not extended after the fact. Without this rule a failing region could be removed
by shrinking the grid.

If the matched Online step is the grid maximum (2.00) in any cell, the result is
reported with an explicit note that the frontier may be truncated, and the grid
may not be extended under this protocol ID.

### Multiplicity

Per-point confidence intervals are descriptive inputs to the deterministic rule.
The decision endpoint is the aggregate coverage count, so no per-point alpha
adjustment is applied. This is stated in advance so it cannot be revisited.

### Power failure is not a pass

If, in more than 25% of the tested Uniform grid points within a scenario, the 95%
CI half-width on `d_AUC` exceeds `delta`, that scenario is declared
**underpowered and inconclusive**. An inconclusive scenario is not a pass and not
a failure; the protocol result is reported as incomplete and the seed count must
be raised under a new protocol ID.

This clause exists so that a noisy experiment cannot produce a pass by having
confidence intervals too wide to exclude anything.

## Pre-declared decision rule

Per scenario: the scenario **passes** if at least **5 of 7** Uniform grid points
are covered.

Overall: the frontier claim receives **GO** if at least **3 of 4** scenarios pass
and no scenario is inconclusive.

Systems feasibility, reported separately: median runtime ratio at the matched
step pairs must be at most 1.5 in every scenario.

## Interpretation limits fixed in advance

A GO supports exactly this statement:

> On the four `ood-v1` scenarios, across a pre-declared step grid and a
> deterministic matching rule, Online RC-OMD attains Uniform Group OMD's success
> AUC within 0.01 while using at most 75% of its absolute distractor KL, at most
> Uniform step-size settings tested.

A GO does **not** support:

- reward improvement at equal nominal step size, which is separately measured and
  not supported ([E1](ERRATA.md));
- generalization to task families outside `ood-v1`;
- any claim about shared-parameter function approximation, where
  `function-approx-v1` returned NO-GO;
- causal credit identification.

A NO-GO means the `ood-v1` GO was dependent on the selected step pair, and the
paper's central Pareto claim must be narrowed accordingly.

## Execution discipline

1. Commit and push this protocol, its config, and the pairing assertion test.
   **Until this happens the protocol is not frozen and results are exploratory.**
2. Record the execution commit in the results document.
3. Run the config once. No parameter changes.
4. Preserve all per-seed histories under `results/pareto_v1/`.
5. Copy the summary and protocol evaluation JSON to `paper/frozen/`, including
   protocol ID, execution commit, config hash, Python and NumPy versions, and
   SHA256 of each generated file.
6. Report every scenario and every grid point, including failures and
   uncovered points.
7. Any later change to grid, margin, bound, or matching rule requires a new
   protocol ID and is post-hoc.

## Prerequisites (met as of this commit)

- `configs/pareto_v1.json` implements the scenarios, methods, and step grid
  above.
- The pairing assertion in "Frozen design" is checked by
  `tests/test_pareto_pairing.py`
  (`test_first_batch_identical_across_methods_and_step_sizes_at_fixed_seed`).
- `experiments/evaluate_pareto_protocol.py` implements the statistical rules
  below and is covered by `tests/test_evaluate_pareto_protocol.py`.
- `experiments/run_reliability_diagnostics.py`'s summary output now includes
  per-seed arrays (`seeds`, `success_auc_per_seed`,
  `cumulative_distractor_kl_per_seed`, `runtime_seconds_per_seed`) so the
  evaluator can form paired differences directly from `summary.json` without
  re-deriving pairing from `history.csv`.

## Command to be run after the protocol commit

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_reliability_diagnostics --config configs/pareto_v1.json
python -m experiments.evaluate_pareto_protocol `
  --config configs/pareto_v1.json `
  --summary results/pareto_v1/summary.json `
  --output results/pareto_v1/protocol_evaluation.json
```

## Result

Executed 2026-08-15 at execution commit `751c3f6540cb3b640ccdbfe3d7e7ea6ae016b950`
(same commit as the protocol freeze; no code changed in between). **Decision:
GO.** All 4 scenarios passed with full 7/7 coverage; systems feasibility
passed in every scenario. Golden record:
`paper/frozen/pareto-v1-2026-08-14.json`. Full per-point breakdown:
`results/pareto_v1/protocol_evaluation.json`. See
[`paper/CLAIMS.md`](../paper/CLAIMS.md) for the claim this licenses and its
stated limits.

`experiments/evaluate_pareto_protocol.py` does not exist yet. It must implement
the statistical rules above and must be committed with this protocol, so that the
decision procedure is fixed in code before any result is produced.
