# Qwen Performance V1: formal definitions

Protocol ID: `qwen-performance-v1` (definitions stage)

Status: **Q1 frozen as of this commit.** The definitions below are closed.
Downstream numbers are uninterpretable without them, so they are fixed before
any model code is written.

## Scope

This is a **performance experiment, not a mechanism validation**. It does not
test `alpha`, the Jacobian estimator, or whether the synthetic mechanism
transfers. No result here is evidence for or against `geometry-v1/-v2/-v3` or
Stage B/B-2 in either direction. See the scope note in `paper/CLAIMS.md`.

## D1. Position

**A position is one token index in the generated completion.** Prompt tokens
are excluded: no action is taken at them, so they have no policy row to move.

Rejected alternatives, recorded so the choice is not silently revisited:

- *Reasoning step.* Requires segmenting chains of thought, which is a
  heuristic that would sit upstream of every number this protocol reports.
- *Whole trajectory.* Collapses the per-position allocation that RWP-OMD exists
  to perform, making the algorithm identical to its baseline by construction.

## D2. Reliability

Let a **group** be the `G` rollouts sampled for one prompt, and `A_i` the
GRPO-normalized scalar advantage of rollout `i`. For token index `t`, let
`S_t = { i : len_i > t }` be the rollouts that reach that index, and
`n_t = |S_t|`.

```
dispersion_t = stdev({ A_i : i in S_t })          (0 when n_t < 2)
raw_t        = [ n_t / (n_t + warmup) ] * [ 1 / (1 + c * dispersion_t) ]
r_t          = max(r_floor, raw_t)
```

`warmup`, `c` and `r_floor` are the language-model analogues of
`warmup_effective_samples`, `confidence_multiplier` and `reliability_floor`.
They are **not** inherited from the synthetic runs; they are selected on the
development split at Q4.

The two factors carry the same meaning as in the synthetic estimator: a token
index reached by few rollouts has little evidence behind it, and one where the
group's outcomes disagree has unreliable credit. Reliability is computed from
the **rollout group alone** and never from the candidate's own updates.

**Known limitation, stated now rather than discovered later.** Tokens are
aligned by index across rollouts, and different rollouts say different things at
the same index. This is a genuinely crude alignment. It is adopted because the
alternatives -- prefix-tree alignment or semantic segmentation -- introduce
machinery whose failure modes would be inseparable from the algorithm's. If
Qwen Performance V1 returns NO-GO, **alignment is a confound and not a refutation
of RWP-OMD**, and the report must say so.

## D3. Mean-one normalization

```
w_t        = r_t + lambda * (1 - r_t)
w_tilde_t  = w_t / mean(w over all generated tokens in the optimization batch)
```

Normalizing over the batch, not per sequence, keeps `lambda` a pure
differential-weighting knob and prevents it from also rescaling the effective
learning rate -- the same reason the synthetic specification normalizes.

## D4. Injection into GRPO

The exponentiated-gradient target boosts only the sampled token:

```
q_t(a) = pi_old,t(a) * exp(eta * A_i * 1[a = a_t]) / Z_t
Z_t    = 1 + pi_old,t(a_t) * (exp(eta * A_i) - 1)
```

By the collapse identity the reliability-weighted objective is a single weighted
cross-entropy against a mixture, `m_t = [r_t q_t + lambda (1 - r_t) pi_old,t] / w_t`,
and because `q_t` differs from `pi_old,t` on one coordinate the mixture is
`pi_old,t` rescaled by two constants:

```
c1 = [ r_t / Z_t                      + lambda (1 - r_t) ] / w_t     (a != a_t)
c2 = [ r_t * exp(eta * A_i) / Z_t     + lambda (1 - r_t) ] / w_t     (a  = a_t)

loss_t = -c1 * sum_a pi_old,t(a) log pi_theta,t(a)
         - (c2 - c1) * pi_old,t(a_t) * log pi_theta,t(a_t)
```

The first term is a full-vocabulary cross-entropy of the frozen `pi_old` against
the trainable policy. **Q3 must measure whether it fits in memory at the frozen
sequence lengths.** If it does not, the fallback is a top-`k` truncation of
`pi_old` with `k` frozen at Q3 and the renormalization stated explicitly; the
truncation is an approximation and must be reported as one.

The total loss is `sum_t w_tilde_t * loss_t / (number of generated tokens)`.

## D5. Uniform baseline

**Exactly RWP-OMD with `r_t = 1` for every token.** Then `w_t = 1`,
`w_tilde_t = 1`, `c1 = 1/Z_t`, `c2 = exp(eta A_i)/Z_t`, and `m_t = q_t`, which
is the ordinary GRPO/OMD update with no reliability weighting.

This nesting is deliberate and load-bearing. It gives Q3 an **exact equivalence
test**: with reliability forced to 1, the two branches must produce identical
losses, gradients and sampled trajectories to numerical precision, mirroring the
synthetic invariant that `r = 1` recovers `ProjectedGroupOMD` to 1e-12. Any
divergence is an implementation defect and a **HALT**, not a result.

The baseline is otherwise tuned in full: its learning rate is swept on the
development split under the same budget as the candidate's, so a GO cannot be
manufactured by under-tuning it.

## D6. Distractor metrics

**Tier 1** is clean GSM8K, scored by exact match on the extracted final answer
under the frozen prompt template and extraction rule of Q2.

**Tier 2** is a distractor-injected variant, frozen before any evaluation.

```
RobustnessDrop = Acc_clean - Acc_distracted        (lower is better)
```

computed on the **same problems** so the pairing is per-item. Auxiliary,
reported but never decisive: answer flip rate, per-item paired
clean/distracted correctness, generation-length change, and answer-token
log-probability change.

**No distractor-KL analogue is defined**, and none may be introduced later. The
synthetic distractor-KL metric depended on knowing which positions are
distractors, which is exactly what a language model does not give us. Tier 2
measures behaviour, not geometry.

## Statistics

Per-item pairing across methods, 95% paired-bootstrap intervals, variation
reported over **both** problem instances and training seeds, per-seed results
always shown alongside aggregates, and never the best run as the headline.

## What Q1 does not settle

- Model identity, revision, tokenizer, LoRA configuration, lengths, prompt
  template and extraction rule -- all Q2.
- Whether the full-vocabulary cross-entropy fits in memory -- Q3.
- `eta`, `lambda`, `warmup`, `c`, `r_floor`, `G`, temperature -- all Q4, on the
  development split only. **`lambda = 3.0` from the synthetic protocols is
  admissible as a grid candidate and may not be adopted as the value.**
