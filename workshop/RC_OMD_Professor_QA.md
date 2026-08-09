# RC-OMD Professor Q&A Preparation

| Likely question | Concise answer | Evidence / boundary |
|---|---|---|
| What is the main contribution? | We separate estimated credit direction from confidence in that estimate, then let reliability control local OMD movement. | The validated claim is limited to controlled tabular, factorized sequence tasks. |
| Is this causal credit assignment? | No. Our running score is an associative reliability proxy, not an estimate of causal responsibility. | Never use “causal credit” in the presentation or paper claim. |
| How is this different from entropy weighting? | Entropy weighting changes the update signal directly. RC-OMD uses a persistent online reliability statistic to control the local step size or KL budget. | Entropy baselines remain related work and diagnostics, not the claimed novelty. |
| Why Online Mirror Descent? | Policies live on probability simplices, and KL-regularized mirror descent gives an interpretable proximity-controlled update. | We do not claim OMD itself is new. |
| Why use normalized success AUC? | It measures success-learning efficiency throughout training rather than only the final policy. | All comparisons use the frozen protocol and ten seeds. |
| What exactly passed OOD? | All four preregistered scenarios passed: AUC deficit ≤0.01 and distractor-KL ratio ≤0.5. | Observed AUC differences were between −0.001758 and +0.000755; KL ratios were 0.189–0.248. |
| Are you using distractor labels during training? | No. Known pivotal and distractor positions are used to evaluate where policy KL was spent in synthetic tasks. | The algorithm receives terminal rewards and online reliability statistics. |
| Why did the primary methods have different base steps? | They were frozen before OOD evaluation to compare a Pareto-matched operating point. | The same-step diagnostic showed lower AUC but only 10.7%–13.4% of Uniform distractor KL; we explicitly report this. |
| Why is the function-approximation result NO-GO? | Shared features project several local policies through the same parameters. Under aliasing, movement at a pivotal position necessarily appears at a distractor. | Transfer passed only 1/3 scenarios; the preregistered rule required 2/3. |
| Did the reliability estimator collapse under feature sharing? | No. Critical reliability stayed above distractor reliability, with top-k precision 0.704–0.725. | The bottleneck is realizable update geometry, not complete estimator collapse. |
| Does the method work with neural networks or LLMs? | We have no evidence for that. The present NO-GO result says we should not promote the algorithm directly to those settings. | No neural policy or language model was tested. |
| What is the next algorithmic step? | Either a reliability-weighted Fisher or natural-gradient metric, or a constrained projection that fits high-reliability targets while limiting low-reliability KL. | Both require a new development phase and a new frozen protocol. |
| What would count as a successful next phase? | Recover the Pareto behavior under controlled shared or nonlinear features without changing the current frozen NO-GO record. | New methods must be evaluated prospectively, not retrofitted to the existing protocol. |

