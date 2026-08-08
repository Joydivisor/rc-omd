# Milestone 3: Bootstrap Reliability-Calibrated OMD

## Method frozen before the main run

For each sampled group, the learner computes a group-relative action-score
estimate at every sequence position. It then resamples the group with replacement
and recomputes those scores. Let `s_k` be the norm of the full-batch score and
`u_k` the norm of its bootstrap uncertainty. RC-OMD v1 uses

```text
q_k = max(0, 1 - z u_k / max(s_k, epsilon))
eta_k = eta_0 [floor + (1 - floor) q_k].
```

The primary run used 32 bootstrap samples, `z = 1.96`, and a reliability floor of
0.1. A global ablation applies `max_k q_k` to every position; local RC-OMD uses
the individual `q_k` values. The policy update is still an exponentiated-gradient
KL-mirror step.

## Main controlled result

Five seeds were evaluated in an entropy-misleading six-step task and a ten-step
task with four pivotal positions. Values below are mean normalized success AUC
and the fraction of cumulative KL spent on known distractor positions.

| Scenario | Method | Success AUC | Distractor KL fraction |
|---|---|---:|---:|
| Entropy misleading | Uniform Group OMD | 0.9582 | 0.0985 |
| Entropy misleading | Entropy-weighted OMD | 0.9438 | 0.3206 |
| Entropy misleading | Global reliability | 0.8917 | 0.0758 |
| Entropy misleading | Local RC-OMD (`z=1.96`) | 0.8842 | 0.0170 |
| Entropy misleading | Oracle credit | 0.9586 | 0.0000 |
| Long sparse credit | Uniform Group OMD | 0.9445 | 0.1091 |
| Long sparse credit | Entropy-weighted OMD | 0.9243 | 0.4809 |
| Long sparse credit | Global reliability | 0.8594 | 0.0828 |
| Long sparse credit | Local RC-OMD (`z=1.96`) | 0.8409 | 0.0206 |
| Long sparse credit | Oracle credit | 0.9447 | 0.0000 |

Local calibration clearly changes where the policy moves: strict calibration
reduces distractor drift by roughly 81%--83% relative to Uniform Group OMD.
However, it is too conservative and loses substantial learning-curve area. The
oracle result is important because it nearly matches Uniform Group OMD's AUC
while spending no KL on distractors. This separates a promising local-update
principle from an inadequate estimator.

## Threshold ablation

The confidence multiplier produces a smooth empirical trade-off.

| Scenario | Variant | Success AUC | Distractor KL fraction |
|---|---|---:|---:|
| Entropy misleading | `z=0.5` | 0.9502 | 0.0669 |
| Entropy misleading | `z=1.0` | 0.9349 | 0.0378 |
| Entropy misleading | `z=1.96` | 0.8842 | 0.0170 |
| Long sparse credit | `z=0.5` | 0.9325 | 0.0819 |
| Long sparse credit | `z=1.0` | 0.9117 | 0.0541 |
| Long sparse credit | `z=1.96` | 0.8409 | 0.0206 |

Increasing the reliability floor from 0.1 to 0.25 at `z=1.0` recovers some AUC
but also increases distractor KL. No tested variant dominates Uniform Group OMD
on both metrics.

## Computational cost

With 32 bootstrap samples, RC-OMD took about 0.66 seconds per seed on the short
task versus 0.09 seconds for Uniform Group OMD. On the longer task it took about
1.4 seconds versus 0.16 seconds. These small absolute times are CPU-only, but the
roughly 7--9x relative overhead would be unacceptable at language-model scale.

## What is established and what is not

Established in these controlled tasks:

1. Bootstrap uncertainty can separate pivotal and distractor positions in a
   large-batch sanity test.
2. Local reliability scaling strongly reduces unnecessary distractor KL.
3. Confidence strictness controls a sample-efficiency--policy-drift trade-off.
4. Exact local credit can remove distractor drift without sacrificing AUC.

Not established:

1. RC-OMD v1 does not outperform Uniform Group OMD overall.
2. Bootstrap reliability is not yet well calibrated in small, sparse groups.
3. The method has not been tested outside controlled sequence tasks.
4. The bootstrap cost is not suitable for an LLM experiment.

## Research decision

Keep local reliability-controlled OMD, but do not promote the current bootstrap
formula as the final algorithm. The next milestone should target an online,
lower-cost reliability estimator and evaluate calibration directly. Candidate
directions are running moments of action-score agreement, empirical-Bernstein
confidence, or periodic bootstrap calibration with cheap updates between
calibration steps.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_reliability_diagnostics --config configs/reliability_diagnostics.json
python -m experiments.run_reliability_diagnostics --config configs/reliability_ablation.json
```

Generated artifacts are written under `results/` and are ignored by Git.
