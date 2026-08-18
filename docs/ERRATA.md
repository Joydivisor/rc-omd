# Errata

Opened 2026-08-14.

This file records corrections to claims made in `README.md`, `docs/MILESTONE_*.md`,
and `paper/CLAIMS.md`. Each entry states the published claim, what the archived
evidence actually supports, and the arithmetic or code location that establishes
the correction.

No milestone document is edited in place. The milestone records stand as written;
this file is the correction layer.

## Audit provenance

Both frozen protocols were regenerated at their execution commits on 2026-08-14
and compared against the published tables. Golden records are in
`paper/frozen/`.

| Protocol | Execution commit | Deterministic metrics | Decision |
|---|---|---|---|
| `ood-v1-2026-08-08` | `d37a5ff` | **reproduce exactly** (24/24, <= 5e-7) | GO, unchanged |
| `function-approx-v1-2026-08-09` | `2c91c69` | **reproduce exactly** (18/18, <= 5e-7) | NO-GO, unchanged |

A recheck at `HEAD` (`f1e33c1`) compared 260 deterministic values in the M5
summary against the execution-commit run: **zero differences**. The only change
is 20 added `cumulative_projection_kl_mean` fields, a schema addition from the
projected-OMD work. Post-freeze commits did not move any published number.

The central empirical results of this project are therefore confirmed
reproducible. Two things are not, and they are recorded as E7 and E8.

**The original artifacts are unrecoverable.** `results/` was never tracked in any
commit in the repository history, and no `summary.json` or
`protocol_evaluation.json` has ever existed in Git. The machine that produced the
published tables is not available. The records in `paper/frozen/` are therefore
labelled `REPRODUCTION`, not originals: they are canonical going forward because
their deterministic metrics match the published values exactly, but they are not
the original byte stream.

## E1. Same-step reward superiority

**Published framing.** `paper/CLAIMS.md` lists "Improved reward at a fixed nominal
step size in every task" under "Not yet supported", which reads as untested.

**Correction.** This was tested, and the tested cells did not support it. The
correct status is *not supported in tested same-step controls*, not *untested*.

**Archived evidence, 12 cells total:**

| Source | Cells | Result |
|---|---|---|
| [M4 matched sweep](MILESTONE_4_ONLINE_RELIABILITY.md) | 2 tasks x 4 base steps = 8 | Uniform AUC higher in all 8 |
| [M5 same-step diagnostic](MILESTONE_5_OOD_RESULTS.md) | 4 OOD scenarios @ step 1.0 | Online AUC lower by 0.0044--0.0069 in all 4 |

In the same cells, Online RC-OMD used substantially less absolute distractor KL
(10.7%--13.4% of Uniform in the M5 same-step diagnostic). The finding is a
consistent Pareto trade-off, not a reward regression alone.

**Wording that is not supported.** "Refuted." Twelve cells across two task
families cannot refute the universal statement. The claim is unsupported in the
tested region; it is not disproven in general.

**Note on provenance.** A wider same-step control (4 OOD scenarios x 3 shared
step sizes) has been referenced in review discussion. That result is not archived
in this repository and is deliberately **not** cited here. It is now known to be
**unrecoverable**, not merely unarchived: `results/` was never tracked in any
commit, so the underlying numbers exist nowhere. If that control is wanted, it
must be re-run under a new protocol ID. The 12 cells above are sufficient and are
fully published.

**Second instance of the same taxonomy error.** `paper/CLAIMS.md` also listed
"Benefits under shared-parameter function approximation" under "Not yet
supported". That hypothesis was tested by protocol
`function-approx-v1-2026-08-09`, which returned **NO-GO** at 1/3 scenarios. Like
E1, it is tested and not supported, not open. `paper/CLAIMS.md` has been
restructured to separate "tested and not supported" from "not yet tested" so that
both cases are stated correctly.

## E2. Distractor-KL reduction: fraction versus absolute

**Published claim.** [M3](MILESTONE_3_BOOTSTRAP_RC_OMD.md) states that strict
calibration "reduces distractor drift by roughly 81%--83% relative to Uniform
Group OMD."

**Correction.** That range is a reduction in the **distractor KL fraction**, the
share of total KL spent on distractor positions. It is not a reduction in
absolute distractor KL. The M3 table column is labelled `Distractor KL fraction`.

Verifying from the published M3 table:

- entropy-misleading: 1 - 0.0170 / 0.0985 = **82.7%**
- long-sparse-credit: 1 - 0.0206 / 0.1091 = **81.1%**

**What absolute reductions are available.** The [M4 table](MILESTONE_4_ONLINE_RELIABILITY.md)
publishes absolute distractor KL and supports absolute statements, but for
**Online RC-OMD**, not for the M3 bootstrap estimator:

- entropy-misleading: 1 - 0.00304 / 0.02440 = **87.5%**
- long-sparse-credit: 1 - 0.00346 / 0.02682 = **87.1%**

**Absolute values for M3, recovered 2026-08-14.** M3 was re-run and its
deterministic metrics reproduce exactly, so the absolute distractor KL that the
milestone document omitted is recoverable after all:

| Scenario | Uniform | Local RC-OMD | Absolute reduction |
|---|---:|---:|---:|
| Entropy misleading | 0.02440 | 0.00247 | **89.9%** |
| Long sparse credit | 0.02682 | 0.00261 | **90.3%** |

These figures are genuine and may be cited, attributed to this reanalysis rather
than to the milestone document. An earlier review concluded they were
underivable; that was correct with respect to the *published table*, which prints
fractions only, but the underlying quantity was always well defined and has now
been reproduced.

With the [E4](#e4-bootstrap-uncertainty-is-computed-outside-the-update-relevant-subspace)
correction applied, the reductions improve to 91.6% and 92.0%. See
`docs/MILESTONE_3_CORRECTED_REANALYSIS.md`.

**Consequence.** The published 81%--83% remains a fraction reduction and must be
labelled as such. Absolute reductions must be quoted from the table above, with
the estimator version stated, because corrected and uncorrected differ.

## E3. Reliability separation under feature aliasing

**Published data.** The [M6 mechanism diagnostic](MILESTONE_6_FUNCTION_APPROX_RESULTS.md)
reports critical and distractor reliability per scenario.

**Correction.** The critical/distractor reliability ratio degrades mildly; it does
not collapse or invert.

| Scenario | Critical | Distractor | Ratio | Top-k precision |
|---|---:|---:|---:|---:|
| Separable | 0.401 | 0.137 | **2.93** | 0.725 |
| Partial aliasing | 0.424 | 0.154 | **2.75** | 0.712 |
| Complete aliasing | 0.432 | 0.167 | **2.59** | 0.704 |

Both reliabilities rise together as aliasing increases, so separation falls only
from 2.93 to 2.59, and top-k precision from 0.725 to 0.704.

**Wording that is not supported.** Any statement that the ratio falls to a value
below 1, such as "2.93 to 0.59". No published table supports it. The distinction
matters for the research conclusion: M6 failed on the **projection**, not on the
**estimator**. The M6 document already states this correctly; the errata exists to
prevent the estimator from being blamed in downstream summaries.

## E4. Bootstrap uncertainty is computed outside the update-relevant subspace

**Defect.** In [`credit_estimators/bootstrap.py`](../credit_estimators/bootstrap.py),
the signal and uncertainty norms are not taken in the same space:

- line 104--105: `signal_norm` is the norm of scores **centered across actions**;
- line 103, 106: `uncertainty_norm` is the norm of the **uncentered** bootstrap
  standard deviation.

The exponentiated-gradient update is invariant to a constant shift applied to all
actions at a position, so the shared component contributes to `uncertainty_norm`
while being excluded from `signal_norm`. The reliability ratio is therefore
inflated in the conservative direction.

**Status: fixed and measured.** The paired re-run is complete. See
`docs/MILESTONE_3_CORRECTED_REANALYSIS.md`.

The correction is a **Pareto improvement** for local RC-OMD: every local variant
gained AUC *and* reduced distractor KL simultaneously.

| Scenario | AUC before | AUC after | KL frac before | KL frac after |
|---|---:|---:|---:|---:|
| Entropy misleading, `z=1.96` | 0.8842 | 0.8992 | 0.0170 | 0.0132 |
| Long sparse credit, `z=1.96` | 0.8409 | 0.8494 | 0.0206 | 0.0156 |

The gain is largest at the strictest confidence multiplier, which is what the
mechanism predicts: inflating `u_k` matters most where `z` is largest.
Non-bootstrap methods are bitwise unchanged, confirming the fix is correctly
scoped.

**The M3 conclusion does not change.** Corrected local RC-OMD still reaches only
0.8992 against Uniform's 0.9582, and 0.8494 against 0.9445. RC-OMD v1 does not
outperform Uniform Group OMD, and estimator calibration remains the bottleneck.
What changes is that part of the conservatism blamed on the estimator was an
artifact of measuring uncertainty outside the update-relevant subspace. The
estimator is better than Milestone 3 reported, and still not good enough.

The global-reliability ablation does **not** inherit the improvement: its AUC
rose but its distractor KL fraction rose too in the entropy-misleading task
(0.0758 to 0.0810), because a single strongest-position confidence applied
everywhere also loosens the distractor positions.

## E5. Method-dependent random seeding in Milestone 2

**Defect.** [`experiments/run_credit_diagnostics.py:102`](../experiments/run_credit_diagnostics.py)
seeds each run with:

```python
rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_index, method_index]))
```

Because `method_index` enters the seed, each method draws a different trajectory
stream at the same nominal seed, so the comparison is unpaired. `method_index` is
the position in the config's method list, so reordering methods in
`configs/credit_diagnostics.json` also changes the numbers.

**Scope.** This affects **Milestone 2 only**.
[`experiments/run_reliability_diagnostics.py:201`](../experiments/run_reliability_diagnostics.py)
uses `np.random.default_rng(seed)` with no method index, so M3 through M6 already
use common random numbers across methods and are unaffected.

**Status: fixed and measured.** See `docs/MILESTONE_2_CORRECTED_REANALYSIS.md`.

**Both M2 conclusions survive with the same sign:**

| Claim | Gap before | Gap after | Verdict |
|---|---:|---:|---|
| Entropy helps when aligned | +0.0028 | +0.0029 | holds |
| Entropy hurts when misaligned | -0.0157 | -0.0148 | holds |

The provisional downgrade previously recorded here is therefore **lifted**: the
aligned-scenario direction is not an artifact of unpaired sampling.

**A sharper form of the defect than first described.** NumPy treats a trailing
zero in a `SeedSequence` entropy list as a no-op, so `[seed, scenario, 0]` and
`[seed, scenario]` are the same stream. The method at index 0 was already on the
stream the correction adopts and reproduces bitwise; only methods at later
indices moved. The defect silently privileged whichever method was listed first
in the config, and the distortion applied to every other method depended on
nothing more principled than list order.

**What is still weak.** The aligned effect is +0.0029 AUC against an across-seed
SD of about 0.0004, with no confidence interval and no pre-declared test. The
direction is real and no longer confounded; the magnitude is not established.
M2 remains exploratory.

## E6. Unreachable clamp

**Status: removed.**

[`credit_estimators/bootstrap.py`](../credit_estimators/bootstrap.py) applied
`reliability = np.minimum(reliability, 1.0)`. With `confidence_multiplier >= 0`
and `uncertainty_norm >= 0` (both enforced at construction), the expression
`1 - z * u / max(s, eps)` never exceeds 1, and the `np.where` false branch yields
0.0. The clamp could not fire.

Removed as a separate change from E4, even though both touch the same function,
so that the numerically inert edit cannot be confused with the one that moves
results. Verified inert: re-running `configs/reliability_diagnostics.json`
compared 140 deterministic values against the pre-removal run and found **0
differences**.

## E7. Published runtime ratios do not reproduce

**Published claim.** `paper/CLAIMS.md` states that the Online RC-OMD "measured CPU
runtime ratio was 1.069--1.107" under the frozen OOD protocol.

**Correction.** Runtime ratios are wall-clock measurements and are
machine-dependent. Regenerating M5 at `d37a5ff` on 2026-08-14 produced:

| Scenario | Published | Regenerated |
|---|---:|---:|
| Dense 2-of-6, tiny group | 1.107 | 1.099 |
| Needle 5-of-5, long horizon | 1.081 | **0.933** |
| Threshold 3-of-5, small group | 1.069 | 1.076 |
| Threshold 4-of-6, three actions | 1.079 | 1.087 |

The needle scenario inverts: Online RC-OMD ran **faster** than Uniform on this
machine, against a published claim that it was 8.1% slower. M6 shows the same
instability (0.999 to 1.040, 1.055 to 1.082, 1.048 to 1.055).

**Consequence.** The interval "1.069--1.107" is a property of one machine on one
day, not of the algorithm, and must not be quoted as a measured constant. The
same applies to the M4 statement that Online RC-OMD "runs within roughly 7--13%
of Uniform Group OMD's CPU time."

**The decision is unaffected.** Every observed ratio remains far below the frozen
systems-feasibility threshold of 1.5, so the PASS holds in both runs. What fails
is the precision of the published figure, not the conclusion drawn from it.

Future protocols should report runtime as an order-of-magnitude feasibility check
with the hardware named, or replace wall-clock with a deterministic proxy such as
operation counts.

## E8. The M5 decision rule was implemented in code after the results existed

**Finding.** `experiments/evaluate_ood_protocol.py` and `tests/test_ood_protocol.py`
were both first added in `9db36ab`, the commit that also added
`docs/MILESTONE_5_OOD_RESULTS.md`. At the protocol commit `d37a5ff`, the Go/No-Go
rule existed only as prose in `docs/MILESTONE_5_OOD_PROTOCOL.md`.

**Assessment.** The prose rule is specific (AUC no more than 0.01 below, absolute
distractor KL at most 50%, at least 3 of 4 scenarios), and the code implements it
faithfully: `MAX_AUC_DEFICIT = 0.01`, `MAX_DISTRACTOR_KL_RATIO = 0.5`,
`REQUIRED_SCENARIO_PASSES = 3`. The evaluator has not been modified since. So
there is no evidence the rule was adjusted to fit the data, and the M5 GO stands.

**But the audit chain has a gap.** Nothing in the repository proves the threshold
constants predate the results, because the file that contains them was created
alongside them. The protocol document carries that burden alone.

**Milestone 6 does not have this gap.** `evaluate_function_approx_protocol.py` was
added in the protocol commit `f779782`, two commits before the results commit
`de43576`. The discipline improved between M5 and M6.

**Forward rule.** The evaluator must be committed with the protocol, before
execution. `docs/PARETO_V1_PROTOCOL.md` already requires this.

## E9. The geometry-v1 mechanism claim does not survive step-size matching

**Published claim.** `paper/CLAIMS.md` states, under the frozen geometry-v1
protocol, that "under partial aliasing RWP-OMD cut distractor KL to 0.575 of
baseline while cutting critical KL only to 0.709 -- selective protection, **not
merely smaller steps**."

**Correction.** The emphasised inference does not follow from that measurement,
and is not supported when the comparison is made against a step-swept uniform
frontier rather than the single baseline step. Re-analysing the frozen
geometry-v1 test seeds (200--219) with a paired uniform sweep gives, under
`partial_feature_aliasing`:

| quantity | RWP-OMD | v1 algorithm |
|---|---|---|
| distractor KL at matched critical KL | 0.8185 [0.8039, 0.8333] | 0.8227 [0.8020, 0.8439] |
| frontier AUC advantage | +0.00033 [-0.00037, +0.00102] | +0.00011 [-0.00046, +0.00068] |

The two algorithms are statistically indistinguishable on allocation, and
RWP-OMD's frontier advantage is null. Under `geometry-v2` on fresh seeds
(500--519) with a finer grid the same scenario measured +0.00067 with a
confidence lower bound of +0.00002, which is **-0.00147 after correcting for
the interpolation bias recorded in E10** -- consistent with the null.

**Consequence.** For `partial_feature_aliasing`, the ratio 0.575 versus 0.709 is
a description of what RWP-OMD did, not evidence that it did something a smaller
uniform step could not. The phrase "not merely smaller steps" is withdrawn for
that scenario. It is **not** withdrawn for `geom_holdout_a` and `geom_holdout_b`,
where the frontier advantage is positive and separates from the v1 algorithm.

**The geometry-v1 GO is unaffected.** Its decision rule compared against a fixed
baseline step and every declared condition was met. The golden record
`paper/frozen/geometry-v1-2026-08-15.json` is accurate as a record of that
protocol and is **not** modified; the protocol itself recorded the fixed-`eta`
confound as its largest open limitation. What is corrected here is a mechanism
sentence in `CLAIMS.md` that read more strongly than the protocol licensed.

## E10. The frontier metric carries a systematic positive bias

**Affected.** The `geometry-v2` primary metric, `Delta_frontier`, and any figure
derived from it.

**Defect.** The metric interpolates the uniform frontier's success AUC at the
candidate's distractor KL, linearly in log KL. Success AUC is **concave** in log
distractor KL, so linear interpolation lies below the true frontier and the
difference is inflated.

**Measurement.** Removing one uniform grid point and interpolating its own AUC
from the remaining points, where the true value is exactly zero by construction,
returns positive in **99.2%** of cases across all eight `geometry-v2` scenarios,
mean **+0.00136**, per-scenario range +0.00092 to +0.00214.

**Consequence.** Effects smaller than roughly +0.002 cannot be distinguished from
this artifact, and a zero decision margin is not safe at that scale. The
`geometry-v2` NO-GO is unaffected: `geom_v2_a3_pure0_homog` measured +0.00494
against a local bias of +0.00096, so the refutation holds on both the raw and the
corrected metric. What the bias does invalidate is any reading of the marginal
positives, `partial_feature_aliasing` in particular.

**Forward rule.** A successor protocol must either declare a margin above the
measured bias, apply the leave-one-out correction as a pre-registered step, or
interpolate on a scale where the frontier is not concave. The leave-one-out test
should be run as a standard control, since it needs no new data.

## E11. The geometry-v3 confirmation criterion was self-contradictory

**Affected.** `docs/GEOMETRY_V3_PROTOCOL.md`, confirmation criterion, and the
verdict in `paper/frozen/geometry-v3-2026-08-19-confirmation.json`.

**Defect.** The frozen criterion required a 95% bootstrap confidence lower bound
**strictly greater than +0.50** while simultaneously requiring the sign of `rho`
to match the discovery set. The selection rule ranks candidates by **absolute**
correlation, so a negative predictor was always reachable. For any such
predictor the two clauses are jointly unsatisfiable, however strong the
correlation. The contradiction is provable from the document alone.

**It fired.** `alpha` was selected at discovery `rho = -0.7868`, and the
confirmation set returned `rho = -0.8859`, 95% interval `[-0.9519, -0.7210]` --
one of the strongest correlations obtainable -- which the literal criterion
scored as NOT CONFIRMED.

**Correction.** The bound is on magnitude in the discovery direction:
`ci_lower > +0.50` when the discovery sign is positive, `ci_upper < -0.50` when
negative. This is the only reading under which the selection and confirmation
rules are mutually consistent. The 0.50 threshold is unchanged.

**This amendment was made after the confirmation subset was executed**, which is
what pre-registration exists to prevent. Three facts bound the discretion
involved: the defect is demonstrable without reference to any measurement; the
threshold did not move; and the observed interval is far from the bound in
either reading. Had the result been marginal, the correct course would have been
to report NOT CONFIRMED and re-run under a corrected protocol. A reader is
entitled to discount the confirmation on this ground, and both the protocol and
the golden record say so in those terms.

**Forward rule.** A confirmation criterion must be stated direction-agnostically
whenever the selection rule ranks by absolute correlation, and a protocol's
selection and decision rules must be checked against each other for joint
satisfiability before freezing. A unit test asserting that a strongly negative
predictor can pass is the cheapest way to catch this class of defect.

## Deferred: corrections that require re-running

These are **not** asserted. They are recorded so that the required experiment is
defined before it is run, rather than after.

1. ~~**Corrected-bootstrap paired re-run.**~~ **Done**, see E4 and
   `docs/MILESTONE_3_CORRECTED_REANALYSIS.md`.
2. **Common-random-number M2 reanalysis.** Remove `method_index` from the seed,
   re-run, and publish as a separate corrected reanalysis. Do not overwrite the
   original M2 table.
3. ~~**Absolute distractor KL for M3.**~~ **Done**, recovered by re-running M3;
   values recorded under E2.

Each must be a separate change with its own record. They must not be combined
into one re-run, because their effects on the published numbers would then be
inseparable.
