# Milestone 2: Controlled Credit Diagnostics

## Question

Does policy entropy identify positions with high causal credit, and what happens
when an OMD optimizer trusts entropy in a setting where this proxy is misleading?

This milestone is diagnostic. It does not claim that the current toy results
establish a generally superior optimizer.

## Controlled setup

Both scenarios use a six-step binary-action task with terminal reward. Only
positions 1 and 4 determine success; the other positions are distractors.

- **Entropy aligned:** pivotal positions start at maximum entropy and distractor
  positions start confident.
- **Entropy misleading:** distractor positions start at maximum entropy, while
  the two pivotal positions start with lower entropy and only 0.2 probability on
  their required actions.

The compared methods are Uniform Group OMD, Entropy-weighted OMD, and
Oracle-credit OMD. The oracle method uses exact on-trajectory counterfactual
credit as a diagnostic reference. It is not a performance upper bound because
all methods use the same untuned base step size.

Configuration: 400 iterations, group size 64, 10 random seeds, and an evaluation
interval of five iterations. The exact configuration is stored in
`configs/credit_diagnostics.json`.

## Preliminary results

| Scenario | Method | Final success | Normalized success AUC |
|---|---|---:|---:|
| Entropy aligned | Uniform Group OMD | 0.9994 | 0.9864 |
| Entropy aligned | Entropy-weighted OMD | 0.9977 | 0.9892 |
| Entropy aligned | Oracle-credit OMD | 0.9994 | 0.9864 |
| Entropy misleading | Uniform Group OMD | 0.9993 | 0.9619 |
| Entropy misleading | Entropy-weighted OMD | 0.9931 | 0.9462 |
| Entropy misleading | Oracle-credit OMD | 0.9994 | 0.9610 |

At initialization, entropy has correlation 0.926 with oracle position importance
in the aligned scenario and -0.775 in the misleading scenario. Its top-k
precision for identifying the two pivotal positions changes from 1.0 to 0.0.

The entropy-weighted method has a slightly larger learning-curve area in the
aligned scenario, but a smaller area in the misleading scenario. All methods
eventually solve these small tasks, so the informative result is the change in
sample efficiency and update allocation, not asymptotic task solvability.

## Decision

Entropy should remain a candidate credit proxy and baseline, not be treated as
ground-truth credit. The next algorithmic milestone will estimate proxy
reliability and use that estimate to control per-position OMD step sizes or KL
budgets. The core hypothesis is that low-reliability credit should produce a
smaller local policy update.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_credit_diagnostics --config configs/credit_diagnostics.json
```

Generated histories, summaries, and figures are written to
`results/credit_diagnostics/` and are intentionally ignored by Git.
