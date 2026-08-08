# Milestone 4: Low-Cost Online Credit Reliability

## Motivation

Bootstrap RC-OMD reduced policy drift on distractor positions but cost roughly
7--9 times as much as Uniform Group OMD. It also produced a conservative
sample-efficiency--drift trade-off. This milestone asks whether credit reliability
can be estimated from persistent agreement across training groups instead of
resampling every group.

## Online estimator

For each training group, we compute the same inverse-propensity group-relative
action scores used by Uniform Group OMD. We maintain exponentially weighted first
and second moments of the centered score at each position and action. The
effective sample size is computed from the sum and squared sum of exponential
weights. Reliability is

```text
q_k = maturity * max(0, 1 - z * standard_error_k / signal_norm_k),
```

where `maturity` increases to one during a short warm-up. The current action score
is updated with a local step

```text
eta_k = eta_0 [floor + (1 - floor) q_k].
```

This requires `O(H A)` state and no extra rollouts or bootstrap resamples.

## Matched estimator comparison

Five seeds were run in the entropy-misleading and long-sparse-credit tasks.
Online variants used base step size 0.5, confidence multiplier 1.0, reliability
floor 0.1, and decay 0.8, 0.9, or 0.95. Error bars in generated Pareto figures
are across-seed standard deviations (`n=5`). Underlying per-seed histories are
stored in the generated CSV files.

With decay 0.9:

| Scenario | Method | AUC | Absolute distractor KL | Runtime/seed |
|---|---|---:|---:|---:|
| Entropy misleading | Uniform, eta=0.5 | 0.9582 | 0.02440 | 0.094 s |
| Entropy misleading | Bootstrap RC, z=0.5 | 0.9502 | 0.01460 | 0.86 s |
| Entropy misleading | Online RC, eta=0.5 | 0.9411 | 0.00304 | 0.105 s |
| Long sparse credit | Uniform, eta=0.5 | 0.9445 | 0.02682 | 0.165 s |
| Long sparse credit | Bootstrap RC, z=0.5 | 0.9325 | 0.01738 | 1.84 s |
| Long sparse credit | Online RC, eta=0.5 | 0.9290 | 0.00346 | 0.18 s |

Online RC-OMD removes most bootstrap cost and suppresses substantially more
distractor drift. Its lower AUC at the same base step suggests that reliable
critical positions also receive an effective step below one.

## Matched step-size sweep

We therefore swept the same base steps for Uniform and Online RC-OMD. This is an
exploratory calibration study, not an independent held-out evaluation.

### Entropy-misleading task

| Base step | Uniform AUC / distractor KL | Online AUC / distractor KL |
|---:|---:|---:|
| 0.50 | 0.9582 / 0.02440 | 0.9411 / 0.00304 |
| 0.75 | 0.9725 / 0.03986 | 0.9565 / 0.00458 |
| 1.00 | 0.9789 / 0.05351 | 0.9648 / 0.00416 |
| 1.25 | 0.9825 / 0.06081 | 0.9698 / 0.00819 |

### Long-sparse-credit task

| Base step | Uniform AUC / distractor KL | Online AUC / distractor KL |
|---:|---:|---:|
| 0.50 | 0.9445 / 0.02682 | 0.9290 / 0.00346 |
| 0.75 | 0.9614 / 0.04635 | 0.9485 / 0.00476 |
| 1.00 | 0.9694 / 0.06045 | 0.9579 / 0.00966 |
| 1.25 | 0.9747 / 0.07827 | 0.9648 / 0.01230 |

At a fixed base step, Uniform OMD has higher AUC. Online RC-OMD moves the Pareto
frontier toward much lower irrelevant drift: for example, Online eta=1.25 is
close to Uniform eta=0.75 in AUC while using roughly 80%--90% less absolute
distractor KL. On the long task, Online eta=1.25 has higher AUC than Uniform
eta=0.75 while using about 73% less distractor KL. These comparisons were selected
after observing the sweep and must be validated on new tasks.

## Reliability diagnostics

At decay 0.9, mean critical reliability exceeded mean distractor reliability in
both tasks, and top-k identification precision was about 0.83 and 0.78. Longer
memory increased ranking precision but slightly reduced early AUC, consistent
with greater estimator lag.

## Limitations

1. The running moments combine data collected under changing policies; they are
   not a stationary confidence interval.
2. The same two environment families were used for algorithm development and
   calibration.
3. The estimator detects persistent association, not causal credit.
4. AUC and distractor KL express different objectives; no universal scalar
   preference is assumed.
5. Runtime measurements are tiny CPU microbenchmarks, not LLM-scale forecasts.

## Decision

Online reliability is a better systems candidate than per-group bootstrap and
should become the primary RC-OMD implementation. The next Go/No-Go test must use
new, predeclared environment structures and group sizes without retuning the
chosen online configuration. The provisional configuration is decay 0.9,
confidence multiplier 1.0, warm-up 8, reliability floor 0.1, with the base step
treated as an ordinary optimizer hyperparameter shared in the comparison grid.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_reliability_diagnostics --config configs/online_reliability_comparison.json
python -m experiments.run_reliability_diagnostics --config configs/online_step_size_ablation.json
```

Generated artifacts, including PNG/PDF Pareto figures and per-seed CSV files, are
written under `results/` and ignored by Git.
