# RC-OMD Workshop — 5-Minute English Speech

## Slide 1 — Group Introduction

Good morning. We are Team PRISM from RL Group 1. Our project asks a narrow optimization question: when a sparse terminal reward gives us a noisy step-level credit estimate, can its reliability control how far Online Mirror Descent moves? Huang coordinates the project, Liu leads theory, Xie builds environments and estimators, Zhong develops algorithms and infrastructure, and Li leads experiments and analysis. Today I will show one preregistered positive result, followed by a deliberately retained negative transfer result.

## Slide 2 — Sparse Verifiable Reward and Credit Assignment

The setting is sparse verifiable reward. A complete trajectory receives one binary score. Group-relative methods form an advantage by subtracting the group mean, which avoids a learned critic. But that one trajectory-level advantage is then broadcast through the whole sequence. This creates an allocation problem: routine and pivotal decisions can move together, and all-correct or all-wrong groups can contain no relative signal. Our project does not claim to discover causal responsibility. It studies whether an online reliability signal can control the geometry of the update under this limited feedback.

## Slide 3 — Reliability-Calibrated OMD

RC-OMD separates two questions. The estimated credit chooses where the policy should move. A reliability score chooses how far that local decision is allowed to move through a step-specific learning rate, or equivalently a local KL budget. In our online variant, reliability is a persistent standardized score updated from grouped rollouts. The controlled environments know which positions are pivotal or distracting, but those labels are used only for evaluation. The algorithm receives the terminal reward and its online reliability statistics. Therefore our positive claim is about movement allocation in a tabular factorized policy, not causal credit recovery.

## Slide 4 — Pre-Registered OOD Positive Result

Before running the OOD study, we froze the tasks, hyperparameters and pass rule. Online RC-OMD had to lose no more than 0.01 normalized success AUC, use at most half the distractor KL of Uniform, and pass at least three of four scenarios. It passed all four. The absolute AUC differences were at most 0.0018, while distractor KL fell to 18.9 to 24.8 percent of Uniform. Runtime overhead was 6.9 to 10.7 percent. This is a Pareto result with different preregistered base steps, not a claim that reliability scaling always improves reward at the same nominal step size.

## Slide 5 — Shared-Parameter NO-GO and Questions

Our first shared-parameter transfer is a NO-GO. With separable linear features, the distractor-KL ratio was 0.193 and the scenario passed. Under partial aliasing it rose to 0.967, and under complete aliasing to 1.176. All three AUC conditions still passed; the failure came entirely from the frozen KL criterion. Diagnostics show that reliability still separates pivotal from distractor positions, but projection couples their parameter updates. Therefore local step-size control is insufficient when the policy cannot realize independent local trust regions. We do not claim neural or LLM effectiveness. We would value guidance on parameter-space geometry, constrained projection, and the next nonlinear benchmark.

