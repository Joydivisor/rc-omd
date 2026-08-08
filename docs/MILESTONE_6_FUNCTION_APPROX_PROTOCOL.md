# Milestone 6: Pre-Registered Shared-Parameter Transfer Protocol

Protocol: `function-approx-v1-2026-08-09`

Status: **frozen before any full protocol execution**.

## Research question

Does the controlled Pareto behavior of Online RC-OMD survive a minimal form of
function approximation in which multiple sequence positions share policy
parameters?

This is a transfer test, not a neural-network or language-model claim.

## Parametric policy and approximate mirror projection

Each position has a fixed feature vector `phi_k`. Policy logits are produced by a
shared linear map:

```text
logits_k = phi_k W,
pi_k = softmax(logits_k).
```

The algorithm first forms the same tabular exponentiated-gradient target used in
the controlled study. It then projects that target into the shared linear family
by minimizing average target-to-parametric KL for 60 fixed gradient steps. Both
primary methods use the same projection routine. The only primary difference is
whether the tabular target uses a uniform or reliability-scaled local step.

## Frozen scenarios

All scenarios have horizon 12, two actions, six pivotal positions, a 5-of-6
terminal threshold reward, 500 updates, and group size 48.

1. `separable_shared_features`: pivotal and distractor positions use disjoint
   feature clusters. Parameters are shared within each class, but reward learning
   need not move distractors.
2. `partial_feature_aliasing`: half of distractor positions share features with
   pivotal positions. Some irrelevant movement is structurally unavoidable.
3. `complete_feature_aliasing_negative_control`: every distractor shares a
   feature with a pivotal position. Local tabular constraints cannot be exactly
   realized after projection; this is a declared stress test.

The tabular Uniform and Online methods are retained as diagnostic references and
do not enter the transfer decision.

## Frozen primary methods

- Projected Uniform Group OMD: base step 0.75.
- Projected Online RC-OMD: base step 1.25, decay 0.9, confidence multiplier 1.0,
  warm-up 8, reliability floor 0.1.
- Both: 60 projection steps, projection learning rate 0.5, zero ridge penalty,
  tolerance `1e-9`, uniform initial policy.

These optimizer settings are transferred from the tabular study rather than
retuned on the function-approximation tasks.

## Frozen metrics and decision

For each scenario, ten seeds are the replication units. Primary metrics are:

- normalized exact-success AUC;
- absolute cumulative KL at known distractor positions;
- cumulative projection KL;
- CPU runtime;
- critical and distractor reliability diagnostics.

A scenario passes if Projected Online RC-OMD is no more than 0.02 below Projected
Uniform in success AUC and uses at most 75% of its absolute distractor KL. The
transfer decision is **GO** if at least two of three scenarios pass. Systems
feasibility passes if the runtime ratio is at most 1.5 in all three scenarios.

The complete-aliasing scenario is expected to weaken the distractor-KL benefit,
but this directional expectation is descriptive and is not an extra pass rule.

## Interpretation boundary

A GO result would support transfer from independent tabular rows to a fixed
shared linear policy. It would not establish transfer to learned representations,
nonlinear networks, contextual MDPs, or language models. A NO-GO result would be
informative: it would show that the current local target plus projection is not a
sufficient implementation of reliability-calibrated geometry under parameter
sharing.

## Reproduction after the protocol commit

```powershell
python -m experiments.run_reliability_diagnostics `
  --config configs/function_approx_preregistered.json

python -m experiments.evaluate_function_approx_protocol `
  --config configs/function_approx_preregistered.json `
  --summary results/function_approx_preregistered/summary.json `
  --output results/function_approx_preregistered/protocol_evaluation.json
```
