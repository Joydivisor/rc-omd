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

## Status

The controlled environment, Uniform Group OMD, Entropy-weighted OMD, and
Oracle-credit OMD are implemented and covered by unit tests. The next milestone
is a reliability estimator and a local trust-region RC-OMD update.
