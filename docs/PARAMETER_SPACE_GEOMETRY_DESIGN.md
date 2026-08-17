# Parameter-Space Reliability Geometry: v2 Design Specification

Status: **specification locked, protocol not frozen.**

The objective, its gradient, the weight-normalization policy, the degeneracy
behaviour, the KL accounting definitions, and the aliasing taxonomy below are
**locked**: they may not change once `geometry-v1` scenarios are run, and any
later change requires a new protocol ID. Still **open**, and deliberately
excluded from this document: development and test scenario definitions, the
`lambda`/`mu` grids, seed lists, the selection rule, and success criteria.
Those belong to `docs/GEOMETRY_V1_PROTOCOL.md`, which does not exist yet.

No implementation exists yet, no experiment has been run, and no claim of
effectiveness is made or implied anywhere in this document.

## 0. What changed from the previous revision

The previous revision of this file proposed a **mixed-direction** objective:
`r_k D_KL(q_k || pi_theta) + lambda (1-r_k) D_KL(pi_theta || pi_old)`. That form
is superseded. It is **not convex** in `theta`: sampling the Hessian of the
all-reverse variant at random points gave a minimum eigenvalue of `-2.69`,
against `-1.6e-07` (numerical zero) for the form adopted below. The mixed and
all-reverse forms were rejected in favour of keeping the projection subproblem
convex, so that reliability geometry is not confounded with non-convex
optimization behaviour.

The problem statement, the mechanism analysis, and the M6 attribution in the
previous revision are unchanged and are restated in condensed form in Section 1.

## 1. The problem this addresses

Under `ProjectedGroupOMD`, position `k`'s policy is `softmax(features[k] @ theta)`.
Two positions with an identical feature row are therefore mapped to the *same*
policy for every `theta` -- identically equal by construction, not merely
correlated. `function-approx-v1-2026-08-09` returned **NO-GO** on 1/3 scenarios
because of this: the M6 mechanism diagnostic shows the reliability *estimator*
survives aliasing (critical/distractor separation degrades only 2.93 -> 2.59,
[E3](ERRATA.md)), while the projection cannot realize independent local trust
regions.

Concretely, in the current code reliability shapes the tabular target
(`projected_omd.py:91-92`, `168-170`) but is absent from the line that actually
arbitrates between coupled positions:

```python
gradient = self.features.T @ (projected_policy - target_policy)   # projected_omd.py:100
```

This is an unweighted fit. When a reliable position and an unreliable position
share a feature row and disagree, nothing in that expression knows which one to
trust. The v2 objective makes that arbitration explicit.

## 2. Algorithm identity

- Name: **Reliability-Weighted Cross-Entropy Projection OMD**
- Short name: **RWP-OMD**
- Protocol/version tag: `geometry-v1`
- Planned class: `ReliabilityWeightedProjectionOMD`, subclassing `ProjectedGroupOMD`

## 3. Locked objective

```
L(theta) =   sum_k  r_k          * D_KL( q_k      || pi_theta^k )
           + lambda * sum_k (1-r_k) * D_KL( pi_old^k || pi_theta^k )
           + (mu/2) * || theta - theta_old ||_F^2
```

Both divergence terms are **forward KL**: the parametric policy `pi_theta` is in
the *second* argument of both. This is the single most important structural
commitment in this document, and Sections 5 and 8 depend on it.

Symbols:

| Symbol | Meaning | Range |
|---|---|---|
| `q_k` | raw OMD target at position `k`, Section 4 | simplex |
| `r_k` | floor-adjusted reliability, `floor + (1-floor)*reliability_k` | `[floor, 1]` |
| `pi_old^k` | policy before this update | simplex |
| `lambda` | protection strength for unreliable positions | `>= 1`, Section 12 |
| `mu` | parameter-space ridge | `>= 0`, Section 12 |

## 4. `q_k` is the RAW OMD target (locked)

```
q_k = softmax( log pi_old^k + eta * action_scores_k )
```

with **no reliability rescaling applied**. Reliability enters the algorithm
exactly once, through `r_k` in the objective. Applying the reliability-scaled
step here as well (as `ProjectedOnlineReliabilityOMD` does at
`projected_omd.py:91-92`) would double-count it: once in the target's step
length and again in the mixture of Section 5. `eta` is the base step size,
unchanged in meaning from every existing method.

## 5. Collapse identity (locked, verified exactly)

A positive combination of forward KLs against fixed references is a single
weighted cross-entropy. Writing `CE(p, pi) = -sum_a p_a log pi_a`:

```
r_k * D_KL(q_k || pi) + lambda (1-r_k) * D_KL(pi_old^k || pi)
      ==  w_k * CE( m_k , pi )  +  const(pi)

      w_k = r_k + lambda * (1 - r_k)                                  (aggregate weight)
      m_k = [ r_k * q_k  +  lambda (1-r_k) * pi_old^k ] / w_k         (arithmetic mixture)
```

Verified numerically: `LHS - RHS` evaluated at six random `pi` was exactly `0.0`
in every case, confirming the residual is constant in `pi`. `m_k` is a valid
distribution (rows sum to 1).

Two consequences that must be stated together, because quoting either alone is
misleading:

- **`lambda` controls two coupled channels**, not one: the mixture ratio inside
  `m_k`, and the aggregate weight `w_k`. Both push the same direction (larger
  `lambda` protects unreliable positions more), so they are treated as one knob.
- **At `lambda = 1`, `w_k == 1` for every position.** Differentiated aggregate
  weighting is switched off, but `m_k = r_k q_k + (1-r_k) pi_old^k` still depends
  on reliability. The precise statement is therefore: *`lambda = 1` disables
  differentiated aggregate weighting while retaining reliability-weighted target
  mixing.* Any fix to the M6 aliasing failure must come from `lambda > 1`;
  `lambda = 1` is a reference point, not a candidate.

## 6. Weight normalization (locked): mean normalization

The objective uses **mean-normalized** weights:

```
w_tilde_k = w_k / ( (1/H) * sum_j w_j )
```

Rationale. `w_k = r_k + lambda(1-r_k)` grows with `lambda`, so an unnormalized
objective would let `lambda` change three things at once: the differential
weighting (intended), the total gradient magnitude, and hence the effective
projection learning rate, the relative strength of `mu`, and the meaning of the
gradient-norm termination tolerance (all unintended). Measured on a random
reliability draw:

| `lambda` | `mean(w)` | `spread(w)` | `mean(w_tilde)` | `spread(w_tilde)` |
|---:|---:|---:|---:|---:|
| 1.0 | 1.000 | 0.000 | 1.000 | 0.000 |
| 2.0 | 1.482 | 0.418 | 1.000 | 0.282 |
| 5.0 | 2.929 | 1.674 | 1.000 | 0.571 |

Mean normalization holds the global scale fixed at 1 while preserving the spread
that carries `lambda`'s intended effect. It leaves the `lambda = 1` reference
point exactly unchanged (spread stays 0), and it preserves convexity because
`w_tilde` depends only on `r`, which is constant during a projection solve.

## 7. Update equations and pseudocode

Per-logit gradient of the collapsed objective is `w_tilde_k * (pi_k - m_k)`, so

```
grad_theta L = F^T [ w_tilde * (Pi - M) ] / H  +  mu * (theta - theta_old)
```

where `F` is `(H, D)`, `Pi` and `M` are `(H, A)`, and `w_tilde` broadcasts over
actions. Verified against central finite differences: max absolute discrepancy
`1.27e-10`.

Relative to the current code this is a one-line change to `projected_omd.py:100`,
from `F.T @ (pi - q)` to `F.T @ (w_tilde[:, None] * (pi - m))`, plus construction
of `m`. The existing tolerance handling, iteration cap, and offset removal carry
over unchanged.

```text
update(trajectories, rewards):
    estimate <- credit_estimator.estimate(...)              # unchanged from v1
    r        <- floor + (1 - floor) * estimate.reliability  # (H,)
    pi_old   <- current policy
    q        <- softmax( log pi_old + eta * estimate.action_scores )   # RAW

    w        <- r + lambda * (1 - r)
    w_tilde  <- w / mean(w)
    m        <- ( r[:,None]*q + lambda*(1-r)[:,None]*pi_old ) / w[:,None]

    theta    <- theta_old                                   # warm start
    for t in 1 .. projection_steps:
        pi   <- softmax(F @ theta)
        g    <- F^T ( w_tilde[:,None] * (pi - m) ) / H  +  mu * (theta - theta_old)
        if ||g|| <= projection_tolerance: break
        theta <- theta - projection_lr * g
        theta <- theta - mean(theta, axis=actions)          # drop softmax-invariant offset
    return stats
```

## 8. Convexity and uniqueness

`CE(m_k, softmax(f_k theta))` equals `-m_k . (f_k theta) + logsumexp(f_k theta)`:
linear plus a convex function of a linear map, hence **convex in `theta`**. A
positive-weighted sum of convex terms plus a convex quadratic is convex, so `L`
is convex for every `mu >= 0` and every `w_tilde > 0`.

Uniqueness needs care because `softmax` is invariant to `theta -> theta + u 1_A^T`
for any `u` in `R^D`, a genuine null direction of the cross-entropy part.

- `mu > 0`: the ridge covers all free parameters, including that null direction,
  so `L` is **strictly convex with a unique minimizer**.
- `mu = 0`: the minimizer is unique only modulo the offset. The offset removal
  already present in `ProjectedGroupOMD` pins it down, but strict convexity is
  not available as a guarantee.

For that reason the `mu` grid must contain at least one strictly positive value
(Section 12), and reproducible projection residuals are asserted as a test
(Section 13) rather than assumed from convexity alone.

## 9. Degeneracy behaviour (locked, verified)

| Condition | `w_k` | `m_k` | Resulting behaviour |
|---|---|---|---|
| `r_k = 1` for all `k` | `1` | `q_k` | **Exactly recovers `ProjectedGroupOMD`** -- the objective becomes today's unweighted forward-KL projection |
| `r_k = 0` for all `k` | `lambda` | `pi_old^k` | Fits the current policy; **zero movement** |
| `lambda = 1` | `1` (uniform) | `r_k q_k + (1-r_k) pi_old^k` | Uniform aggregate weight, reliability retained in the target (Section 5) |
| one-hot features | -- | -- | `pi_k = m_k` exactly; the update is an arithmetic reliability interpolation |

The `r = 1` row is a property the all-reverse variant did **not** have, and is a
further reason for the Section 3 choice: v2 contains the current projected
baseline as an exact special case.

## 10. Arithmetic vs geometric mixing, and what cannot be inherited

Under one-hot features the update is `pi_k = m_k`, an **arithmetic**
interpolation between `q_k` and `pi_old^k`. Tabular Online RC-OMD instead
produces a **geometric** interpolation, `pi propto q^r * pi_old^(1-r)`. These
agree only to first order in `eta` and diverge at the step sizes actually used:

| `eta` | max abs policy gap (arithmetic vs geometric) |
|---:|---:|
| 0.25 | 0.0025 |
| 0.50 | 0.0095 |
| 1.00 | 0.031 |
| **1.25** (Online RC-OMD's frozen step) | **0.045 - 0.054** |
| 2.00 | 0.133 - 0.136 |

**Consequence, and it is binding:** RWP-OMD is a different algorithm from Online
RC-OMD in the tabular limit. It therefore **cannot inherit the empirical
conclusions of `pareto-v1-2026-08-14` or `ood-v1-2026-08-08`**. The three M6
scenarios become a genuine in-distribution validation benchmark, not a formal
sanity check, and the separable scenario in particular is a real test rather
than a foregone pass. This must appear in the interpretation limits of
`geometry-v1`.

## 11. KL accounting (locked definitions)

Three quantities that are routinely conflated are given distinct names here.
All follow the existing repository convention that the post-update distribution
occupies the first argument, matching `row_kl` in
`experiments/run_reliability_diagnostics.py`.

| Name | Definition | Meaning |
|---|---|---|
| **Target KL** | `D_KL( m_k || pi_old^k )` | movement the update *requests* at position `k`, before the parametric family constrains it |
| **Realized KL** | `D_KL( pi_new^k || pi_old^k )` | movement actually delivered; this is what the existing distractor/pivotal KL metrics measure |
| **Projection residual** | `D_KL( m_k || pi_new^k )` | how far the family fell short of the request; matches the existing `projection_kl` statistic |

The gap between target and realized KL at distractor positions is the direct
quantitative expression of the M6 bottleneck, and both must be logged per
position.

## 12. Hyperparameter policy (policy locked, values open)

**`lambda`.** Grid restricted to `lambda >= 1`.
- `lambda = 1` is the mandatory reference point (uniform aggregate weight).
- `lambda > 1` is where any M6 fix must originate (Section 5).
- `lambda < 1` is **excluded from the search**: it assigns *less* weight to less
  reliable positions, dragging them harder, which is the wrong direction. At
  most one `lambda < 1` point may be run as a labelled negative check, never as
  a selection candidate.

**`mu`.** Grid must contain at least one strictly positive value, for the
uniqueness guarantee in Section 8. `mu = 0` is admissible as a comparison point.

**Selection discipline.** `lambda` and `mu` are selected **only** on development
scenarios (Section 14), and are frozen before any test scenario is executed.
Selecting them on the M6 scenarios would repeat exactly the defect that
`pareto-v1-2026-08-14` was created to remove from `ood-v1`. The selection rule
must be deterministic, including tie-breaks, and must be committed before the
development sweep is run.

## 13. Required tests and numerical invariants

To be implemented alongside the algorithm, before any scenario is run:

1. **Simplex validity** -- policy rows sum to 1 and stay strictly positive at
   every iteration.
2. **Action-shift invariance** -- adding a constant to all action scores at a
   position leaves the update unchanged.
3. **Aggregation identity** -- the two-term objective equals `w_k CE(m_k, pi)`
   up to a `pi`-independent constant (Section 5).
4. **Analytic gradient** -- matches central finite differences (Section 7).
5. **`r = 0` gives zero movement** -- policy and parameters unchanged.
6. **`r = 1` recovers `ProjectedGroupOMD`** -- bitwise or to tight tolerance
   against the existing class on the same inputs.
7. **One-hot reduction** -- with one-hot features the update equals the
   arithmetic mixture `m_k` to projection tolerance.
8. **Complete-aliasing negative control** -- with `alpha = 1` features, realized
   critical KL equals realized distractor KL exactly (Section 14, Proposition).
9. **Reproducible projection residual** -- identical inputs give identical
   residuals across runs.
10. **Convexity spot-check** -- sampled Hessian minimum eigenvalue is
    non-negative to numerical tolerance.

## 14. Aliasing taxonomy (locked, data-generating definition)

Positions `k` and `k'` are **tied** when their feature rows are equal,
`F[k] == F[k']`. Because `pi_k = softmax(F[k] theta)`, tied positions have
identical policies for every `theta`. Let `G` be the partition of positions into
tie-groups, and for group `g` let `c(g)` and `d(g)` count its critical and
distractor members.

**Aliasing index:**

```
alpha(F, critical) = 1 - (1/H) * sum_{g in G} | c(g) - d(g) |        in [0, 1]
```

equivalently `alpha = (1/H) * sum_g 2*min(c(g), d(g))`.

| Regime | Condition | Meaning |
|---|---|---|
| **Separable** | `alpha = 0`, i.e. `min(c,d) = 0` for every group | no critical position shares a feature row with a distractor |
| **Partial aliasing** | `0 < alpha < 1` | some groups mix critical and distractor positions, not all balanced |
| **Complete aliasing** | `alpha = 1`, i.e. `c(g) = d(g)` for every group | every position is in a balanced mixed group |

Computed on the existing M6 configs, this reproduces the established taxonomy
exactly:

| M6 scenario | tie-groups | `alpha` |
|---|---|---:|
| `separable_shared_features` | `c2/d0` x3, `c0/d2` x3 | **0.000** |
| `partial_feature_aliasing` | `c2/d1` x3, `c0/d3` x1 | **0.500** |
| `complete_feature_aliasing_negative_control` | `c2/d2` x3 | **1.000** |

**"Incomplete aliasing" and "partial aliasing" denote the same regime.** The
term *incomplete* is retired; scenarios are described as "partial, `alpha = x`".

**Proposition (negative-control guarantee).** If `alpha = 1` then realized
critical KL equals realized distractor KL exactly, for any update within this
parametric family. *Proof:* `alpha = 1` forces `c(g) = d(g)` for every group.
Tied positions have identical policies before and after any update, so all
members of `g` share one per-position realized KL `kappa_g`. Critical mass is
`sum_g c(g) kappa_g`, distractor mass is `sum_g d(g) kappa_g`, and these are
equal term by term. QED.

This is a geometric fact about the parameterization, not an algorithmic
shortcoming. `geometry-v1` must therefore treat complete aliasing **strictly as
a negative control**: equal critical and distractor KL is the *expected correct
outcome*, must not be counted as a success, and an apparent improvement there
should be investigated as a bug.

**Limitation.** `alpha` is defined by exact feature-row equality, which suffices
for the one-hot feature matrices used throughout this repository. Graded or
near-aliasing (rows close but unequal) would require a subspace-angle
generalization and is out of scope for `geometry-v1`.

## 15. Scenario partitioning (policy locked, scenarios open)

- **Development set:** new shared-feature scenarios, spanning a range of `alpha`
  including at least one separable, one partial, and one complete case, with
  horizons, critical positions, and tie-group structures distinct from M6. Used
  for `lambda`/`mu` selection and for nothing else.
- **In-distribution diagnostic set:** the three existing M6 scenarios. Reported
  in full, never used for tuning. These are seen data: M6's outcome on them is
  already known.
- **Held-out set:** **at least two** further shared-feature scenarios, unused for
  tuning and distinct from both sets above. One is insufficient -- a single
  scenario lets within-scenario noise decide the protocol outcome.

Positive success criteria are defined on partial aliasing and on the held-out
set. Separable serves as a regression check, complete aliasing as the negative
control of Section 14.

## 16. Manuscript wording constraints

- `D_KL(pi_old || pi_theta)` must **not** be described as a Bregman proximal
  step or a mirror-descent proximity term. The Bregman divergence of negative
  entropy is `D_KL(pi_new || pi_old)`, the opposite direction. Acceptable
  descriptions: *forward cross-entropy projection*, or *parameter-space fitting
  objective*. The mirror-descent geometry of this method lives in how `q_k` is
  constructed (Section 4), which is unchanged exponentiated gradient; it does
  not extend to the projection.
- RWP-OMD results may not be described as inheriting, extending, or confirming
  the `ood-v1` or `pareto-v1` conclusions (Section 10).
- No effectiveness claim of any kind may be made until `geometry-v1` has been
  frozen and executed.

## 17. What remains open

Everything required to freeze `geometry-v1`, none of which is decided here:
concrete development, diagnostic, and held-out scenario definitions; the
`lambda` and `mu` grids; projection step count and learning rate; seed lists;
the deterministic selection rule and its tie-breaks; baselines and negative
controls; per-scenario and overall success criteria; and the evaluator that
implements them in code before any result exists.

Until that document is committed, GSM8K and every other language-model
experiment remain out of scope.
