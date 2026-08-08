# Research Charter

## Working title

Adaptive Credit-Weighted Online Mirror Descent for Sparse Verifiable Reward Reinforcement Learning

## Research question

Can the reliability of step-level credit estimates be used to control the magnitude of Online Mirror Descent updates under sparse verifiable rewards?

## Core quantities

For step `k`, define:

- `c_k`: oracle or ground-truth counterfactual credit;
- `c_hat_k`: estimated credit available to the learner;
- `sigma_hat_k`: uncertainty estimate for `c_hat_k`.

## Hypotheses

1. Common credit proxies can be systematically wrong in controlled environments, including high-entropy distractors and low-entropy pivotal actions.
2. Credit-estimation error can produce harmful policy drift when the update geometry is not reliability-aware.
3. A reliability-calibrated OMD update can improve the reward--KL-drift--sample-efficiency trade-off under matched compute budgets.

## Minimum baselines

1. Uniform group-relative OMD.
2. Entropy-weighted OMD.
3. Global adaptive-KL OMD.
4. Oracle-credit OMD.

## Primary metrics

- Success-rate area under the learning curve.
- Samples to reach a fixed success threshold.
- Credit sign accuracy and calibration error.
- Harmful-update rate.
- Per-step and total KL drift.
- Runtime and additional rollout cost.

## Scope boundary

The controlled tabular/MLP study is the core project. A small-model RLVR experiment is optional and will proceed only after the controlled study passes its Go/No-Go criteria.
