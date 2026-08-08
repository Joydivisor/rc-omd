# Milestone 5: Pre-Registered OOD Results

Protocol: `ood-v1-2026-08-08`

Execution commit: `d37a5ff`

The protocol, tasks, hyperparameters, metrics, and decision rule were committed
and pushed before this run. No task or parameter was changed after results were
observed. All four scenarios and all ten seeds are reported.

## Frozen primary comparison

- Uniform Group OMD: base step 0.75.
- Online RC-OMD: base step 1.25, reliability decay 0.9, confidence multiplier
  1.0, warm-up 8, reliability floor 0.1.

A scenario was pre-declared to pass if Online RC-OMD was no more than 0.01 below
Uniform in normalized success AUC and used at most 50% of Uniform's absolute
cumulative distractor KL. Generalization required at least three of four passes.

## Primary results

Across-seed SD uses the seed as the replication unit (`n=10`).

| OOD scenario | Uniform AUC | Online AUC | AUC difference | Uniform distractor KL | Online distractor KL | KL ratio | Runtime ratio | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Dense 2-of-6, tiny group | 0.992182 | 0.990865 | -0.001317 | 0.074726 | 0.018519 | 0.248 | 1.107 | Yes |
| Needle 5-of-5, long horizon | 0.980026 | 0.980782 | +0.000755 | 0.158354 | 0.037777 | 0.239 | 1.081 | Yes |
| Threshold 3-of-5, small group | 0.992195 | 0.990567 | -0.001628 | 0.126358 | 0.023867 | 0.189 | 1.069 | Yes |
| Threshold 4-of-6, three actions | 0.989446 | 0.987688 | -0.001758 | 0.087531 | 0.017376 | 0.199 | 1.079 | Yes |

Result: **4/4 scenarios pass; pre-registered decision = GO.**

All four runtime ratios are below the frozen 1.5 threshold, so the separate
systems-feasibility decision is also **PASS**.

## Same-step diagnostic

At base step 1.0 for both methods, Online RC-OMD was 0.0044--0.0069 lower in AUC
but used only 10.7%--13.4% as much absolute distractor KL. This confirms that the
primary result is a Pareto trade-off, not a claim that reliability scaling always
improves reward at identical nominal step size.

## Interpretation

The OOD result supports the narrow controlled claim:

> Persistent online credit reliability can be used to allocate mirror-descent
> policy movement away from known distractor decisions while retaining nearly
> all success-learning efficiency under a pre-declared Pareto-matched comparison.

The evidence is stronger than the development result because it covers a new
threshold reward rule, two- and three-action policies, group sizes 24--96,
horizons 10--14, and initial exact-success probabilities from 0.03125 to 0.58.

## What this does not establish

1. The tasks remain tabular, factorized sequence environments with known
   distractor positions available only for evaluation.
2. Running score persistence is associative rather than a causal-credit proof.
3. The primary pair uses different base steps selected on development tasks.
4. Ten seeds quantify variation in these simulations but do not establish
   universality across RL environments.
5. No neural policy, function approximation, or language model has been tested.

## Reproducibility

Run the frozen experiment:

```powershell
python -m experiments.run_reliability_diagnostics --config configs/ood_preregistered.json
```

Apply the frozen decision rule:

```powershell
python -m experiments.evaluate_ood_protocol `
  --config configs/ood_preregistered.json `
  --summary results/ood_preregistered/summary.json `
  --output results/ood_preregistered/protocol_evaluation.json
```

The generated directory contains per-seed history, summary JSON, machine-readable
protocol evaluation, success curves, and PNG/PDF Pareto figures. Generated data
are ignored by Git.

## Next decision

The controlled-study Go criterion is satisfied. The next phase should package the
controlled contribution into a paper draft and add one modest function-
approximation experiment. A language-model experiment remains optional and
should not begin until the theoretical statement and neural experiment protocol
are frozen.
