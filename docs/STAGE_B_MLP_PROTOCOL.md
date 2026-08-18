# Stage B Protocol: does `alpha` survive non-linearity?

Protocol ID: `stage-b-mlp-2026-08-19`

Status: **document frozen as of this commit.** Architecture, baseline training
configuration, the continuous definition of `alpha`, its measurement layer,
temporal averaging, normalization, the scenario family, the split, and the
decision rules below are closed.

Freeze stages:

1. **Document frozen (this commit).** Protocol only.
2. **Structure frozen.** Model, generator, configs, evaluator, tests committed;
   the discovery/confirmation split committed **before any scenario runs**.
3. **Calibration confirmed.** The continuous `alpha` estimator is shown to
   reproduce the discrete `alpha` on linear one-hot scenarios, and the grid
   coverage re-check is committed.

Only after stage 3 may the confirmation subset be executed.

## Purpose

This is **not** a test of whether RWP-OMD still runs under a non-linear policy.
It is a test of one claim:

> `geometry-v3` confirmed that `alpha` predicts RWP-OMD's frontier advantage at
> `rho = -0.8859` on held-out scenarios. Every result supporting that rests on
> assumption **(M1)**, a softmax-**linear** policy over one-hot tie-group
> features. Does `alpha` retain predictive power once (M1) is dropped?

The stake is the Qwen pre-flight gate. If `alpha` predicts under non-linearity,
the gate is reinstated as an **evidence-backed candidate**. If it fails, direct
progression to Qwen is **prohibited** and mechanistic investigation resumes.

## Why (M1) is not a technicality

Dropping linearity breaks three things the earlier protocols relied on:

- The forward-KL objective is **no longer convex**, so the projection has no
  unique optimum.
- The projection becomes **inexact** -- gradient descent on a non-convex
  surface, not a solve.
- Tie-groups become **emergent and time-varying** rather than declared and
  fixed, so `alpha` must be *measured* rather than read off the configuration.

Any of the three could destroy the correlation independently.

## Architecture (frozen)

Per-position logits are produced by a **shared** non-linear map of the
scenario's existing feature vector:

```
logits_k = W2 @ tanh(W1 @ f_k + b1) + b2
```

- `f_k` is the scenario feature row, exactly as in `geometry-v1..v3`.
- Hidden width **32**, single hidden layer, `tanh`.
- `W1, b1, W2, b2` are shared across all positions; there are no
  position-private parameters.
- Initialization: `W1 ~ N(0, 1/sqrt(fan_in))`, `b1 = 0`, **`W2 = 0`**, `b2 = 0`,
  seeded by the run seed.

  **Amended before any code was written.** The original wording drew `W2` from
  the same normal, which produces a non-uniform initial policy. Every scenario
  in this repository declares a uniform `initial_policy`, and the projection
  base class rejects anything else, so the original initialization was
  unusable. Zeroing the output layer makes the initial policy exactly uniform,
  which also matches the linear model's zero-weight initialization and so keeps
  the MLP and (M1) comparisons starting from the same policy. `W1` gradients
  are zero at the first projection step and become non-zero once `W2` moves,
  which is the ordinary behaviour of a zero-initialized output layer and not a
  training defect.
- The softmax-linear model of (M1) is the special case obtained by deleting the
  hidden layer, which is what makes the comparison meaningful.

Parameter sharing is the point: it is what allows movement on one position to
drag another, which is what `alpha` is about.

## Feature-noise axis (frozen)

At exactly one-hot features the tie-groups stay exact even through the MLP,
because identical inputs give identical outputs. That case tests non-linear
*optimization* while holding structure fixed. To also test the continuous
estimator, each scenario is instantiated at three noise levels:

```
f_k(eps) = onehot(g(k)) + eps * z_k ,   z_k ~ N(0, I) drawn once per
                                        (scenario, eps) at a fixed seed
eps in {0.0, 0.10, 0.30}
```

`eps = 0` reproduces the linear structure exactly and is the calibration
anchor. `eps > 0` blurs tie-group boundaries, so `alpha` becomes genuinely
continuous and must be estimated from gradients.

Noise vectors are frozen at generation and stored in the config; they are not
resampled per seed, so `alpha` is a property of the scenario, not of the run.

## Continuous `alpha` (frozen)

Let `s_k = +1` for critical positions and `-1` for distractors. Let `G_k` be the
gradient of position `k`'s logit vector with respect to **all** trainable
parameters, flattened. Define the **normalized, non-negative Gram matrix**

```
S_jk = max(0, <G_j, G_k> / (||G_j|| * ||G_k||))
```

and

```
m_k    = sum_j S_jk                     (effective tie-group size at k)
b_k    = sum_j S_jk * s_j               (signed imbalance of k's neighbourhood)
alpha  = 1 - (1/H) * sum_k |b_k| / m_k
```

**This reduces exactly to the discrete definition.** Under one-hot features
`S_jk = 1` iff `j, k` share a tie-group and `0` otherwise, so `m_k = m_g`,
`b_k = c(g) - d(g)`, and the sum telescopes to `sum_g |c(g) - d(g)|`. Verified
numerically over all 51 `geometry-v3` profiles at `H = 8, 12, 16`: maximum
absolute deviation **1.1e-16**.

Frozen measurement choices:

- **Layer.** All trainable parameters (`W1, b1, W2, b2`) jointly. Not a single
  layer: restricting to one would measure a projection of the geometry that
  actually governs the update.
- **Normalization.** Cosine, i.e. gradients unit-normalized before the inner
  product. Magnitudes are absorbed by the step size and are not structural.
- **Clipping.** Negative cosines are clipped to zero. Anti-alignment is not
  shared membership, and leaving it signed would let two opposed positions
  cancel a third's imbalance. Irrelevant at `eps = 0`, where `S` is already
  `0/1`.
- **Temporal averaging.** `alpha` is computed at every evaluation checkpoint and
  **arithmetically averaged over checkpoints**. `alpha(0)` and `alpha(T)` are
  recorded separately as diagnostics but are **not** the response variable.
- **Arm.** `alpha` is computed **exclusively from the Uniform baseline arm**,
  averaged over that arm's seeds. RWP-OMD enters the definition of `alpha`
  nowhere. This is the non-circularity guarantee: the predictor is a property of
  the task and the architecture, measurable before the candidate algorithm is
  ever run.

## Metric and matching (frozen)

Unchanged from `geometry-v3`, including the **E10** leave-one-out bias
correction as a pre-registered step and the rank-based bracketing with log-KL
interpolation of `geometry-v2`. The `eta` grid, `lambda* = 3.0`, `mu* = 0.0`,
and the 2-of-20 dropped-seed limit are inherited unchanged.

The grid floor is re-checked under the MLP at stage 3, because a non-linear
policy may reach different KL ranges than the linear one. If coverage fails, the
floor is lowered **before** any confirmation scenario runs.

## Scenarios, split, and seeds (frozen)

- **24 base scenarios** sampled by the `geometry-v3` generator under the same
  constraints, with **no profile reused** from `geometry-v1`, `-v2`, or `-v3`.
- Each instantiated at all three `eps`, giving **72 scenario-instances**.
- **Split by base scenario, not by instance**: 14 base scenarios (42 instances)
  to discovery, 10 base scenarios (30 instances) to confirmation, by
  `numpy.random.default_rng(20260821)`. Splitting by base scenario prevents the
  same structure appearing on both sides at different noise levels, which would
  leak.
- Evaluation seeds **700-719** for discovery, **800-819** for confirmation,
  disjoint from each other and from every prior protocol.
- One complete-aliasing control instance per `eps`, held **outside** the family
  and the split, as amended in `geometry-v3`.

## Decision rules

**Calibration gate (stage 3, blocking).** The estimator applied to the
**linear** parameterization at `eps = 0` must match the discrete `alpha` to
within `1e-6` on every scenario. Failure means the estimator is wrong and halts
the protocol; it is not a result about `alpha`.

**Amended before any scenario was executed.** The gate originally required the
measured `alpha` to match the discrete `alpha` under the **MLP** at `eps = 0`.
That is false by construction, and implementation showed it immediately: with
one-hot features the MLP produces identical *policies* within a tie-group, but
its per-position *Jacobians* are not orthogonal across groups. Two mechanisms
cause this and neither is a defect. The output bias `b2` has derivative
`delta_{a,a'}` at every position, identical for all `k`. The hidden activations
`h_k = tanh(W1 f_k + b1)` differ across groups but are not orthogonal. Measured
cross-group cosine similarity is **0.45 to 0.60** on a representative
configuration where the linear model gives exactly `0`.

The gate therefore validates the *estimator*, against the parameterization where
the discrete answer is defined, and the MLP measurement is the experiment rather
than a check. Restricting the Jacobian to a parameter subset chosen to recover
orthogonality was rejected: it contradicts the frozen "all trainable parameters"
choice and would be tuning the instrument toward a desired reading.

This makes the test **more** faithful to the Qwen case, not less. On a
transformer there is no discrete `alpha` to recover either, and `alpha` will
have to be measured exactly as it is here.

**Discovery.** Spearman `rho` between measured `alpha` and the bias-corrected
frontier advantage across the 42 discovery instances. Reported, not decisive.

**Confirmation, executed once on the 30 held-out instances.** Direction-agnostic,
per **E11**:

- **CONFIRMED** iff `sign(rho)` matches `geometry-v3`'s established negative
  direction **and** the 95% bootstrap interval oriented by that sign excludes
  magnitude `0.50`, i.e. `ci_upper < -0.50`. 10000 resamples at
  `numpy.random.default_rng(20260822)`.
- **NOT CONFIRMED** otherwise.

The threshold and the direction are both inherited from `geometry-v3` rather
than chosen here, so neither can be tuned to this data.

**Secondary, reported not decisive.** `rho` computed within each `eps` level
separately, to show whether any correlation is carried by the noise axis rather
than by structure.

## Consequences, fixed in advance

- **CONFIRMED** -> the Qwen pre-flight gate is reinstated as an
  **evidence-backed candidate gate**. Stage D proceeds: resource pre-checks and
  transformer `alpha` measurement.
- **NOT CONFIRMED** -> **direct progression to Qwen is prohibited.** Mechanistic
  investigation resumes on synthetic scenarios. The language-model phase does
  not proceed on the strength of a predictor that failed the one test standing
  between a linear toy and a transformer.

## What this protocol cannot establish

- **Nothing about transformers.** An MLP over fixed features is not an attention
  model over learned representations. A pass here licenses *measuring* `alpha`
  on Qwen; it does not predict what will be found.
- **Nothing causal.** `alpha` remains a correlate.
- **Nothing about the `alpha` / `critical_mass_in_pure_groups` tie.** That margin
  was 0.0024 in `geometry-v3` and is not resolved here; only `alpha` is carried
  forward, as frozen.
- **Nothing about learned representations.** Features are fixed inputs; only the
  map above them is non-linear.

## Archiving

- Raw output: `results/stage_b_mlp/` (gitignored).
- Golden records: `paper/frozen/stage-b-mlp-2026-08-19-discovery.json` and
  `-confirmation.json`, each with protocol ID, execution commit, config and
  summary SHA256, the frozen split, seeds, and per-instance values.
- `paper/CLAIMS.md` and `docs/ERRATA.md` updated in a **separate** commit.
