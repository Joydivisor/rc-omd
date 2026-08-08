# Milestone 6: Shared-Parameter Transfer Results

Protocol: `function-approx-v1-2026-08-09`

Protocol commit: `f779782`

Execution commit: `2c91c69`

The execution commit differs only by an instrumentation fix that exposes
`last_reward_std` for projected reliability diagnostics. The first execution at
the protocol commit produced identical policies, AUC, and KL values; its missing
reliability summaries motivated the diagnostic-only fix.

## Frozen decision

Each scenario passed if Projected Online RC-OMD was no more than 0.02 below
Projected Uniform in normalized success AUC and used at most 75% of its absolute
cumulative distractor KL. Transfer required at least two of three scenario
passes. Systems feasibility required a runtime ratio no greater than 1.5 in all
three scenarios.

Result: **1/3 scenarios pass; transfer decision = NO-GO.**

Result: **3/3 runtime checks pass; systems feasibility = PASS.**

## Primary results

Across-seed SD uses the seed as replication unit (`n=10`).

| Scenario | Uniform AUC | Online AUC | AUC difference | Uniform distractor KL | Online distractor KL | KL ratio | Runtime ratio | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Separable shared features | 0.982550 | 0.980943 | -0.001608 | 0.073491 | 0.014174 | 0.193 | 0.999 | Yes |
| Partial feature aliasing | 0.977562 | 0.977033 | -0.000529 | 0.137077 | 0.132543 | 0.967 | 1.055 | No |
| Complete aliasing negative control | 0.971513 | 0.972722 | +0.001209 | 0.178261 | 0.209638 | 1.176 | 1.048 | No |

All AUC conditions passed. The transfer failure came entirely from the declared
distractor-KL condition.

## Mechanism diagnostic

The reliability estimator did not simply collapse under feature sharing:

| Scenario | Critical reliability | Distractor reliability | Top-k precision |
|---|---:|---:|---:|
| Separable | 0.401 | 0.137 | 0.725 |
| Partial aliasing | 0.424 | 0.154 | 0.712 |
| Complete aliasing | 0.432 | 0.167 | 0.704 |

Despite this separation, the shared projection couples positions with the same
feature. In the complete-aliasing task, every policy movement at a pivotal
position necessarily appears at a paired distractor, so critical and distractor
KL are exactly equal for each projected method. Online RC-OMD uses a larger base
step to recover learning speed; after projection, this creates more distractor
KL than Projected Uniform.

The result identifies a sharper bottleneck:

> Reliability estimation remains informative, but local reliability scales are
> not sufficient when the parametric projection cannot realize independent
> local trust regions.

## Interpretation

The tabular Pareto behavior transfers to shared linear features only when the
feature geometry separates pivotal and distractor decisions. It degrades under
partial aliasing and reverses under complete aliasing. Therefore the current
algorithm should not be promoted directly to a neural or language-model test.

The next method question is how to express reliability in parameter space. Two
candidate directions are:

1. a reliability-weighted Fisher or natural-gradient metric that accounts for
   cross-position coupling; or
2. a constrained projection that explicitly minimizes movement on low-
   reliability decisions while fitting high-reliability OMD targets.

These are new algorithms and require a new development phase followed by a new
frozen protocol. The present NO-GO protocol must remain unchanged.

## Reproduction

```powershell
python -m experiments.run_reliability_diagnostics `
  --config configs/function_approx_preregistered.json

python -m experiments.evaluate_function_approx_protocol `
  --config configs/function_approx_preregistered.json `
  --summary results/function_approx_preregistered/summary.json `
  --output results/function_approx_preregistered/protocol_evaluation.json
```

Generated histories, summaries, figures, and protocol evaluation are under
`results/function_approx_preregistered/` and are ignored by Git.
