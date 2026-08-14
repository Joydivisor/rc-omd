# Parameter-Space Reliability Geometry: Problem Formalization and Direction

Status: **draft, problem formalization and direction selection only.** This is
not a frozen design. It contains no pseudocode, no fixed hyperparameters, no
unit test list, and no freezing protocol. Nothing here licenses a claim of
effectiveness, and no language-model or GSM8K experiment may start from this
document alone. See "What this document is not" at the end.

## 1. Why this document exists

`function-approx-v1-2026-08-09` returned **NO-GO** on 1/3 scenarios: the
tabular Pareto behavior of Online RC-OMD (small AUC cost, large distractor-KL
reduction) held under separable shared features and did not hold under partial
or complete feature aliasing (`docs/MILESTONE_6_FUNCTION_APPROX_RESULTS.md`).
`pareto-v1-2026-08-14` (GO; `paper/frozen/pareto-v1-2026-08-14.json`) has since
shown the tabular result is robust to step-size selection, which rules out "we
just picked a bad step pair" as the explanation for anything -- it sharpens
the question rather than answering it. The M6 failure is a mechanism problem,
not a step-size problem, and this document starts the next development phase
the M6 results doc calls for.

## 2. The mechanism, precisely

Under `ProjectedGroupOMD` (`algorithms/projected_omd.py`), the policy at
position `k` is `softmax(features[k] @ weights)`. Two positions with an
identical feature row are therefore mapped to the *exact same policy* --
not approximately coupled, identically equal, by construction. In the M6
complete-aliasing negative control every pivotal position shares its feature
row with exactly one distractor position, so this is not an edge case in that
scenario, it is the entire scenario.

The mechanism diagnostic in M6 confirms the reliability *estimator* is not
the problem: critical/distractor reliability separation only degrades mildly
under aliasing (2.93 to 2.59 ratio, `E3` in `docs/ERRATA.md`). The failure is
downstream of estimation, in how a reliability scalar per position is turned
into an update.

Trace it through the current code. `ProjectedOnlineReliabilityOMD.update`
computes a reliability estimate and turns it into per-position scales,
`local_scales = reliability_floor + (1 - reliability_floor) * reliability`
(`algorithms/projected_omd.py:168-170`). Those scales enter the tabular OMD
target:

```python
target_logits = np.log(np.maximum(old_policy, self.min_probability))
target_logits += self.step_size * scales[:, None] * scores      # projected_omd.py:91-92
```

So a low-reliability position gets a smaller, more conservative *target*.
But the projection step that follows -- the part that actually resolves what
happens when two positions with different targets share one parametric
output -- does not see reliability at all:

```python
gradient = self.features.T @ (projected_policy - target_policy)   # projected_omd.py:100
```

This is an unweighted least-squares fit of the shared parameters to every
position's target simultaneously. Reliability shaped the targets going in,
but the arbitration between a reliable position's target and an unreliable
position's target, when they are forced to share one output, is left to
whatever plain least-squares does with the residuals. Nothing in that line
knows which of the two positions is trustworthy.

This is the sharper form of the bottleneck stated informally in the M6
results doc ("local reliability scales are not sufficient when the
parametric projection cannot realize independent local trust regions"): the
scales are not *absent* from the pipeline, they are absent from exactly the
one step where coupling is resolved.

## 3. Problem statement

> Under shared parameters, when two or more positions with different
> per-position reliability are forced to share a parametric output because
> they share features, how should the shared update arbitrate between the
> high-reliability positions' targets and the low-reliability positions'
> preference to not move -- as an explicit, principled part of the fitting
> objective, rather than as an implicit residual of an unweighted
> least-squares fit?

Constraints on any candidate answer, taken directly from what M6 already
established and must not be re-litigated:

- It must degrade gracefully to the existing tabular result when features are
  one-hot (no coupling): recovering `ProjectedGroupOMD`'s current behavior
  (or better) in the separable case is a floor, not a target.
- It must not assume the reliability estimator needs fixing. E3 already shows
  it does not collapse under aliasing.
- Under complete aliasing, a critical and its paired distractor position
  literally cannot move independently -- this is a geometric fact of the
  parameterization, not a defect to be engineered away. **The negative
  control's outcome (equal critical/distractor KL under complete aliasing) is
  a fact to be preserved and explained, not a bug to be hidden.** A v2
  algorithm that made critical and distractor KL diverge under complete
  aliasing would be wrong, not improved.

## 4. Primary direction: reliability-weighted projection objective

Replace the current unweighted projection loss with an explicit two-term
objective per projection step:

```
min_theta  sum_k  r_k     * D_KL( q_k || pi_theta_k )
         + lambda * sum_k (1 - r_k) * D_KL( pi_theta_k || pi_old_k )
         + mu * ||theta - theta_old||^2
```

where `q_k` is the existing tabular OMD target (`target_policy` in
`_apply_action_scores`, already computed), `r_k` is the existing reliability
estimate, and `pi_theta_k = softmax(features[k] @ theta)`.

Read against the current code, this is a modification of one gradient
computation, not a new algorithm family:

- Term 1, weighted by `r_k`, is what the existing gradient
  `features.T @ (projected_policy - target_policy)` already approximates,
  except currently unweighted (every position contributes with weight 1
  regardless of `r_k`).
- Term 2 is new: an explicit pull toward `pi_old_k` at low-reliability
  positions, so a distractor sharing a feature with a critical position can
  push back against being dragged along, proportionally to how unreliable it
  is.
- Term 3 (`mu` ridge toward `theta_old`) already exists as
  `projection_ridge * (self.weights - old_weights)`.

At the coupled position pair from Section 2 (shared feature, one critical,
one distractor), the two positions' terms now compete explicitly in the same
objective instead of being pre-baked into targets that are then fit blindly:
the critical position pulls toward its `q_k` with weight `r_k`, the
distractor resists movement with weight `lambda * (1 - r_k')`. What the
shared output actually does becomes a stated arbitration, not an
unexamined by-product.

**Degeneracy checks (informal, to be made precise and tested in Phase 5):**

- `r_k = 1` for all `k`: term 2 vanishes, this should reduce to (at most) the
  current unweighted fit -- i.e. plain tabular-target fitting.
- `r_k = 0` for all `k`: term 1 vanishes, term 2 pulls everything toward
  `pi_old`, so the update should collapse toward no movement. This is the
  formal version of "low-confidence positions produce no extra movement when
  reliability equals zero," already listed as a required test in the
  Phase-5 test plan below.
- Complete aliasing, one critical `r_k` high and its paired distractor `r_k'`
  low: the shared output should sit somewhere between `q_k` and `pi_old_k'`,
  with position determined by `r_k`, `r_k'`, and `lambda` -- not silently
  equal to the unweighted fit's residual split. Whether it should still
  produce equal critical/distractor KL (per the negative-control constraint
  above) or whether "equal KL" was an artifact of the *unweighted* objective
  specifically is an open question this document does not resolve.

## 5. Alternative direction (deferred)

Reliability-weighted Fisher/natural-gradient geometry: embed `r_k` into the
Fisher metric over `theta` directly, rather than into a KL-fitting objective
against tabular targets. This would account for cross-position coupling
through the Fisher information matrix's off-diagonal structure instead of
through an explicit two-term loss. Higher theoretical ceiling (it does not
require the two-stage "tabular target, then project" structure at all), but
substantially higher implementation and validation cost, and no existing code
in this repository is close to it. Treated as a second candidate, not
pursued further here.

## 6. What is not decided yet

This document formalizes the problem and selects a primary direction. It
does **not** fix:

- The value or freezing procedure for `lambda` and `mu`, or whether they
  should be scenario-dependent.
- Whether `q_k` should remain exactly the current `target_policy` construction
  or change now that its downstream use is changing.
- The exact algorithm name, version number, and its relationship to
  `ProjectedGroupOMD` (subclass vs. replacement).
- Pseudocode, hyperparameter determination procedure, or a fixed unit test
  suite -- drafted informally above as degeneracy checks, to be formalized.
- A `geometry-v1` protocol, baselines, negative controls, or pre-registered
  success criteria.

These belong to the still-open **Phase 5** (design specification) and
**Phase 6** (freezing protocol) of the project's stated sequence, and must
produce `docs/PARAMETER_SPACE_GEOMETRY_DESIGN.md`'s frozen successor (or a
revision of this file to frozen status), draft source, unit tests, a
development config, and a draft `geometry-v1` protocol before any claim of
effectiveness is made.

## What this document is not

It is not a frozen design. It is not evidence the primary direction works --
no code implementing it exists yet, and no experiment has been run. It does
not authorize starting GSM8K or any other language-model experiment. Per
`docs/RESEARCH_CHARTER.md` and the discipline established by
`docs/PARETO_V1_PROTOCOL.md`, a design becomes actionable only once its
pseudocode, hyperparameters, tests, and success criteria are committed
together, before execution.
