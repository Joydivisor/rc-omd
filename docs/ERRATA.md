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

**Consequence.** Any absolute distractor-KL reduction attributed to the M3
bootstrap estimator requires the archived per-seed absolute values, which are not
available. Figures near 90% attributed to M3 should not be used. The M4 figures
above may be used, attributed to Online RC-OMD.

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

**Status.** The defect is confirmed by reading the source. Its magnitude is **not**
established, because M3 cannot be re-run here.

**What may be said now.** M3 reliability estimates may have been more conservative
than the update geometry justifies, because uncertainty was measured partly
outside the action-difference subspace that the update acts on.

**What may not be said.** That the M3 estimator conclusion is invalidated, or that
corrected reliability would change the M3 trade-off. Both require the paired
re-run recorded under "Deferred".

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

**Consequence for the M2 conclusion.** The two M2 findings are not equally robust
to this defect. From the published M2 table:

- entropy-aligned: entropy-weighted 0.9892 vs uniform 0.9864, gap **0.0028**
- entropy-misleading: entropy-weighted 0.9462 vs uniform 0.9619, gap **-0.0157**

The misleading-scenario effect is roughly 5.6x the aligned-scenario effect. The
"entropy hurts when misaligned" direction is unlikely to be an artifact of
unpaired sampling. The "entropy helps when aligned" direction rests on a 0.0028
AUC gap measured without common random numbers and should be treated as
provisional until re-run.

This asymmetry matters because the aligned-scenario result is the weaker half of
the claim that motivates the whole reliability programme.

## E6. Unreachable clamp

[`credit_estimators/bootstrap.py:118`](../credit_estimators/bootstrap.py) applies
`reliability = np.minimum(reliability, 1.0)`. With `confidence_multiplier >= 0`
and `uncertainty_norm >= 0` (both enforced), `1 - z * u / max(s, eps)` never
exceeds 1, and the `np.where` false branch yields 0.0. The clamp is dead code.
Cosmetic; no numerical effect.

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

## Deferred: corrections that require re-running

These are **not** asserted. They are recorded so that the required experiment is
defined before it is run, rather than after.

1. **Corrected-bootstrap paired re-run.** Center the bootstrap scores across
   actions before taking the standard deviation, then re-run M3 primary, the `z`
   ablation, the reliability-floor ablation, the global/local comparison, runtime,
   and calibration diagnostics. Report old versus corrected as a paired
   comparison. State explicitly whether the M3 trade-off conclusion changes.
2. **Common-random-number M2 reanalysis.** Remove `method_index` from the seed,
   re-run, and publish as a separate corrected reanalysis. Do not overwrite the
   original M2 table.
3. **Absolute distractor KL for M3.** Recoverable only by re-running M3 and
   recording absolute values alongside fractions.

Each must be a separate change with its own record. They must not be combined
into one re-run, because their effects on the published numbers would then be
inseparable.
