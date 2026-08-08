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

## Status

The repository is being prepared for the Cambridge short-term research project. The first milestone is a controlled sequence MDP with computable ground-truth step credit and verified OMD baselines.
