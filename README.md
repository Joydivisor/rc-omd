# RC-OMD

Research code for:

**Adaptive Credit-Weighted Online Mirror Descent for Sparse Verifiable Reward Reinforcement Learning**

The project studies whether the reliability of step-level credit estimates can be used to control the magnitude of Online Mirror Descent updates under sparse, delayed, verifiable rewards.

## Current research focus

We separate three quantities:

- `c_k`: ground-truth or oracle step credit in controlled environments;
- `c_hat_k`: an estimated step credit;
- `sigma_hat_k`: uncertainty in that estimate.

The main research question is whether uncertainty-aware credit estimation can improve the stability and sample efficiency of mirror-descent policy updates.

## Repository layout

```text
algorithms/          OMD baselines and candidate RC-OMD updates
credit_estimators/   Entropy, bootstrap, agreement, and counterfactual estimators
environments/        Controlled sparse-reward environments with oracle credit
experiments/          Reproducible experiment entry points
configs/              Versioned experiment configurations
tests/                Unit and sanity tests
docs/                 Research charter, literature matrix, and decision log
paper/                Working manuscript and figures
proposal/             Submitted project proposal source and QA materials
outputs/              Selected project deliverables
```

## Reproducibility principles

Every experiment should record its configuration, random seed, Git commit, software versions, hardware, runtime, and output path. Large checkpoints and raw experiment dumps should remain outside Git and be referenced by metadata.

## Quick start

The first milestone uses only NumPy and Matplotlib:

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_sequence_baseline --config configs/sequence_baseline.json
python -m experiments.run_credit_diagnostics --config configs/credit_diagnostics.json
python -m experiments.run_reliability_diagnostics --config configs/reliability_diagnostics.json
python -m experiments.run_reliability_diagnostics --config configs/reliability_ablation.json
python -m experiments.run_reliability_diagnostics --config configs/online_reliability_comparison.json
python -m experiments.run_reliability_diagnostics --config configs/online_step_size_ablation.json
python -m experiments.run_reliability_diagnostics --config configs/ood_preregistered.json
```

The experiment writes a CSV history, JSON summary, and learning-curve figure to
`results/sequence_baseline/`. The results directory is intentionally ignored by
Git so that generated artifacts do not enter the source history.

## Milestone 1: controlled sequence baseline

The initial environment has a terminal binary reward and a configurable subset
of pivotal positions. Distractor positions do not affect reward. Because the
environment is fully specified, it provides exact on-trajectory counterfactual
credit of the form `Q(s_k, a_k) - V(s_k)`. The Uniform Group OMD baseline
broadcasts each trajectory's group-relative advantage across all positions and
applies an exponentiated-gradient KL-mirror update.

## Milestone 2: entropy is a proxy, not credit

Three OMD variants now share the same exponentiated-gradient implementation:

- Uniform Group OMD broadcasts group-relative trajectory advantages;
- Entropy-weighted OMD allocates more update mass to high-entropy positions;
- Oracle-credit OMD uses exact counterfactual step credit as a diagnostic
  reference (not a guaranteed performance upper bound).

The controlled experiment contains an entropy-aligned scenario and a deliberately
misleading scenario with high-entropy distractors and lower-entropy pivotal
positions. Ten-seed results show that entropy weighting improves early learning
when entropy is aligned with oracle importance, but reduces sample efficiency
when entropy is negatively correlated with oracle importance. This motivates the
next step: estimate whether a credit proxy is reliable before allowing it to
control the local OMD update.

See `docs/MILESTONE_2_CREDIT_DIAGNOSTICS.md` for the exact setup and preliminary
measurements.

## Milestone 3: bootstrap reliability and local OMD geometry

RC-OMD v1 bootstraps each group-relative action-score estimate and computes a
position-level confidence shrinkage factor. That factor controls the local OMD
step size. A global-reliability ablation applies the strongest position-level
confidence to every position, isolating local geometry from batch-level
adaptation.

The first result is a trade-off rather than a dominance claim. Strict confidence
thresholds substantially reduce policy KL spent on known distractor positions,
but slow learning. A looser threshold recovers much of the sample efficiency at
the cost of admitting more distractor drift. The oracle-credit diagnostic shows
that accurate local credit can preserve learning speed while eliminating
distractor updates, so estimator calibration is the current bottleneck.

See `docs/MILESTONE_3_BOOTSTRAP_RC_OMD.md` for results, limitations, and the next
research decision.

## Milestone 4: low-cost online reliability

The bootstrap estimator is replaced by exponentially weighted running moments of
the group-relative action scores. Persistent directions receive larger local
steps, while inconsistent finite-sample directions are suppressed. This requires
one action-score computation per group and `O(H A)` state.

In the controlled tasks, Online RC-OMD runs within roughly 7--13% of Uniform
Group OMD's CPU time and uses much less absolute distractor KL. A matched
step-size sweep shows a Pareto trade-off rather than universal dominance:
Uniform OMD is faster at the same base step size, while Online RC-OMD achieves
similar AUC with substantially less distractor drift at a larger base step.

See `docs/MILESTONE_4_ONLINE_RELIABILITY.md` for the complete measurements and
scope limitations.

## Milestone 5: pre-registered OOD validation

Four previously unseen tasks were frozen and committed before execution. They
change the reward from an all-match rule to threshold rules, include two and
three actions, vary horizon and group size, and span sparse to dense initial
success probabilities. The pre-declared Online-vs-Uniform comparison passed all
four scenarios: Online RC-OMD stayed within 0.0018 AUC of Uniform, used only
18.9%--24.8% as much absolute distractor KL, and added 6.9%--10.7% runtime.

See `docs/MILESTONE_5_OOD_PROTOCOL.md` for the frozen protocol and
`docs/MILESTONE_5_OOD_RESULTS.md` for the complete result and limitations.

## Status

The controlled environment suite, diagnostic baselines, bootstrap RC-OMD, and
low-cost Online RC-OMD are implemented and covered by unit tests. Pre-registered
OOD validation passed its frozen Go criterion. A working paper draft is now
available under `paper/`. The pre-registered shared-linear-policy transfer test
returned NO-GO: reliability calibration preserved its Pareto behavior with
separable features but not under partial or complete feature aliasing. The next
algorithmic milestone is therefore parameter-space reliability geometry, not a
premature language-model experiment.
