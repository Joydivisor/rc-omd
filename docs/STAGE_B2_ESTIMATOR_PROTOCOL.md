# Stage B-2 Protocol: can any estimator measure aliasing under non-linearity?

Protocol ID: `stage-b2-estimator-2026-08-19`

Status: **document frozen as of this commit.** The estimator menu, the
calibration gate, the selection rule, the confirmation set, and the decision
rules below are closed.

Freeze stages:

1. **Document frozen (this commit).** Protocol only.
2. **Structure frozen.** Estimators, the fresh confirmation family, the
   evaluator, and tests committed. The confirmation family is generated and its
   sweep executed **before** any estimator is selected.
3. **Estimator frozen.** The variant selected on the Stage B discovery
   instances is committed together with the full selection record.

Only after stage 3 may the confirmation set be evaluated.

## Purpose

`stage-b-mlp-2026-08-19` returned NOT CONFIRMED: the Jacobian cosine estimator
of `alpha` does not predict the frontier advantage under a non-linear policy
(`rho = +0.2089`, CI `[-0.1705, +0.5329]`). Two facts survived that null and
together define this protocol.

The **frontier advantage itself survives non-linearity** -- 40 of 42 discovery
and 26 of 30 confirmation instances positive. And post-hoc, the **discrete**
aliasing index still correlated with it under the MLP while the estimator did
not, which suggests the failure lies in the instrument rather than in the
quantity.

That suggestion is a lead, not a finding: it was computed after the outcome and
its confirmation interval included zero. This protocol tests it.

> Does any estimator in a pre-declared menu recover a measurement of aliasing
> that predicts the frontier advantage under non-linearity?

The stake is unchanged. On a transformer there is no discrete `alpha` to fall
back on, so an estimator that works is a precondition for the language-model
phase, not a convenience.

## A suspected repeat of E4

[E4](ERRATA.md) records that bootstrap uncertainty was once computed "partly
outside the update-relevant action-difference subspace." The Stage B estimator
may repeat that error class: it measures the geometry of **logits**, whereas
what governs the update is the geometry of **parameter movement under the KL
metric**. Under a linear head the two coincide, which is precisely why the
Stage B calibration gate passed at 1.11e-16 and certified nothing about the
non-linear case.

Menu entries 4 and 5 below are the direct tests of that suspicion.

## Estimator menu (closed)

Each entry is a fixed, deterministic transform of the per-position Jacobian
`J_k = d logits_k / d theta` with **no free parameters**, so selecting among
them is not fitting. Nothing may be added after results are seen.

Let `pi_k` be the position's policy row and `1` the all-ones action vector.

| # | name | transform applied before cosine similarity |
|---:|---|---|
| 1 | `jacobian_cosine` | none -- the **Stage B estimator, retained as the refuted control** |
| 2 | `centered` | subtract the across-position mean Jacobian |
| 3 | `no_bias` | drop the `b1` and `b2` blocks |
| 4 | `action_difference` | centre logits across actions, `J_k <- (I - 1 1^T / A) J_k` |
| 5 | `fisher` | weight by the softmax metric, `J_k <- (diag(pi_k) - pi_k pi_k^T) J_k` |
| 6 | `centered_action_difference` | 4 then 2 |

Entry 1 is retained deliberately, on the `geometry-v3` principle: if the
selection procedure ranks the already-refuted estimator first, the procedure is
overfitting and that must be reported.

Entries 4 and 5 encode the E4 suspicion. Entry 4 removes the component to which
the softmax is invariant; entry 5 replaces the Euclidean logit geometry with the
metric that actually governs KL movement. Entry 2 removes the constant
contribution of the output bias, whose derivative is identical at every position
and therefore inflates every similarity uniformly.

`alpha` is computed from the resulting Gram matrix by the unchanged Stage B
formula, on the **Uniform baseline arm only**, checkpoint-averaged then
seed-averaged.

## Calibration gate (stage 2, blocking, per variant)

**Monotone agreement, not exact reproduction.** On the linear head, where the
discrete aliasing index is ground truth, a variant must be a monotone function
of it: for every pair of unperturbed Stage B instances with **different**
discrete `alpha`, the measured values must order the same way.

```
for all i, j with discrete[i] < discrete[j]:  measured[i] <= measured[j]
```

Pairs tied in the discrete index impose no constraint. A variant with any
discordant non-tied pair is **disqualified before selection** and may not be
chosen whatever its correlation with the advantage.

**Amended before selection.** The gate originally read
`spearman(measured, discrete) == 1.0`. That is unusable here: the discrete
index is heavily tied -- 9 distinct values across the 25 unperturbed instances,
with `alpha = 0.5` occurring 9 times -- and Spearman penalises the arbitrary
ordering a continuous estimator imposes *within* a tie group. Measured on the
menu, every variant scored 0.975 to 0.993 and all six were disqualified,
including `jacobian_cosine`, which reproduces the discrete index to better than
`1e-9` and therefore cannot genuinely disagree with it. All six in fact have
**zero discordant pairs among the 249 non-tied pairs**.

The pairwise form tests the property actually required -- that the estimator
orders scenarios correctly wherever the ground truth expresses an order -- and
is insensitive to tie-breaking noise. The amendment was made before any variant
was selected and before the confirmation set was evaluated.

**This gate differs deliberately from Stage B's.** Stage B required exact
numerical reproduction of the discrete index. Copying that here would be wrong:
centring and Fisher weighting change the *values* of the cosines even on a
linear head while preserving their *ordering*, so an exactness gate would
disqualify the very transforms this protocol exists to test. The estimator's job
is to rank scenarios -- every downstream use is a rank correlation -- so rank
agreement is the property that must hold, and exactness is not.

A variant that passes rank agreement but not exactness is therefore admissible,
and the record must state which variants were exact and which merely
rank-faithful.

## Data

**Selection uses the 42 Stage B discovery instances**, which are already spent
for this question. Their `delta_corrected` values are inherited unchanged from
`results/stage_b_mlp/summary.json`; only `alpha` is recomputed per variant, so
selection costs no new sweep.

**Confirmation uses a fresh family**: 10 new base scenarios under the
`geometry-v3` constraints, no tie-group profile reused from `geometry-v1`,
`-v2`, `-v3` or Stage B, each at the three Stage B noise levels, giving **30
instances**. Evaluation seeds **1000-1019**, disjoint from every prior protocol.

The Stage B **confirmation** instances are **not reused**. They were consumed by
the Stage B test and using them again would make this protocol's confirmation a
second look at the same data.

The confirmation sweep is executed at stage 2, before any estimator is selected,
so that selection cannot be influenced by it and no sweep is run after the
choice is known.

## Selection rule (Stage B discovery instances only)

1. Disqualify any variant failing the calibration gate.
2. For each survivor compute Spearman `rho` against `delta_corrected` over the
   42 instances.
3. Rank by `abs(rho)` descending, ties broken by menu order ascending.
4. Select the top variant **only if** `abs(rho) >= 0.50`. Otherwise the outcome
   is **NO ESTIMATOR FOUND**, the confirmation set is not evaluated, and the
   language-model pre-flight screen is recorded as having no instrument.

The threshold matches the confirmation magnitude bound rather than
`geometry-v3`'s 0.70: a variant that cannot reach 0.50 in selection cannot
plausibly clear 0.50 on held-out data, and setting selection above confirmation
would reject candidates the confirmation test could still pass.

## Confirmation criterion (executed once)

Direction-agnostic, per [E11](ERRATA.md), with sign and threshold inherited
rather than chosen here:

- **CONFIRMED** iff `sign(rho)` is negative, matching the direction established
  by `geometry-v3`, **and** `ci_upper < -0.50` on a 95% bootstrap interval with
  10000 resamples at `numpy.random.default_rng(20260824)`.
- **NOT CONFIRMED** otherwise.

## Consequences, fixed in advance

- **CONFIRMED** -> the aliasing geometry is measurable under non-linearity by
  the selected estimator. The Qwen pre-flight gate is reinstated as an
  **evidence-backed candidate gate** and Stage D proceeds.
- **NO ESTIMATOR FOUND or NOT CONFIRMED** -> the failure is **not** localised to
  the instrument, and the E4-style account is wrong or incomplete. Direct
  progression to Qwen remains **prohibited**. The honest reading becomes that
  aliasing geometry is not measurable from parameter gradients in this family,
  and the language-model phase does not proceed on a screen that does not exist.

## What this protocol cannot establish

- **Nothing about transformers.** An MLP over fixed features is not attention
  over learned representations.
- **Nothing causal.** A confirmed estimator measures a correlate.
- **Nothing that rescues Stage B.** Stage B's NOT CONFIRMED stands for the
  estimator it tested. A success here is a new result about a different
  estimator, not a reinterpretation of that one.
- **Nothing about `alpha` versus `critical_mass_in_pure_groups`.** That
  `geometry-v3` tie, margin 0.0024, remains unresolved.

## Archiving

- Raw output: `results/stage_b2/` (gitignored).
- Golden records: `paper/frozen/stage-b2-estimator-2026-08-19-selection.json`
  and `-confirmation.json`, each carrying protocol ID, execution commit, config
  and summary SHA256, the calibration table for every menu entry, and
  per-instance values.
- `paper/CLAIMS.md` and `docs/ERRATA.md` updated in a **separate** commit.
