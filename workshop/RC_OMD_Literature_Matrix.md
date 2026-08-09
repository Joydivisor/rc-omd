# One-Page Literature Matrix

| Work | Main problem | Signal / granularity | Optimization mechanism | Relevance and remaining boundary |
|---|---|---|---|---|
| Beck and Teboulle (2003), *Mirror Descent and Nonlinear Projected Subgradient Methods* | First-order optimization beyond Euclidean geometry | Gradient / decision level | Bregman-proximal mirror step | Supplies the classical OMD geometry; it does not address sparse RL credit. |
| Tomar *et al.* (2020), *Mirror Descent Policy Optimization* | Stable policy improvement | Policy-level advantage | Divergence-controlled policy update | Connects mirror descent to RL policy optimization; uses no step-credit reliability calibration. |
| Shao *et al.* (2024), *DeepSeekMath* | Critic-free RL for mathematical reasoning | Group-relative sequence advantage | GRPO-style clipped policy optimization | Motivates grouped verifiable rewards; broadcasts a sequence-level signal. |
| He *et al.* (2026), *Polarity-Entropy Analysis / EAPO* | Token-level credit collapse in RLVR | Token entropy and polarity | Entropy-aware token weighting | Shows that uniform token treatment is problematic; entropy is not established as causal credit. |
| Xie *et al.* (2026), *ACPO* | Fine-grained adaptive credit | Surrogate entropy / token level | Adaptive credit-weighted policy optimization | Closely related empirical credit weighting; differs from RC-OMD’s reliability-controlled local geometry. |
| Zhang (2026), *From Reasoning to Agentic: Credit Assignment in RL for LLMs* | Organizes the rapidly growing credit-assignment literature | Trajectory, step, token, agent turn | Survey across methods | Establishes the broader open problem; does not validate our controlled optimizer claim. |
| **RC-OMD (this project)** | How aggressively to trust an imperfect credit signal | Persistent online reliability / step level | Reliability-scaled local OMD step or KL budget | OOD GO in tabular factorized tasks; shared-linear transfer NO-GO under feature aliasing; no neural or LLM claim. |

## References

- Beck, A. and Teboulle, M. (2003) ‘Mirror descent and nonlinear projected subgradient methods for convex optimization’, *Operations Research Letters*, 31(3), pp. 167–175.
- He, Y. *et al.* (2026) ‘Rethinking token-level credit assignment in RLVR: A polarity-entropy analysis’, arXiv:2604.11056.
- Shao, Z. *et al.* (2024) ‘DeepSeekMath: Pushing the limits of mathematical reasoning in open language models’, arXiv:2402.03300.
- Tomar, M. *et al.* (2020) ‘Mirror Descent Policy Optimization’, arXiv:2005.09814.
- Xie, Z. *et al.* (2026) ‘ACPO: Adaptive credit policy optimization via fine-grained surrogate entropy’, arXiv:2607.03126.
- Zhang, C. (2026) ‘From reasoning to agentic: Credit assignment in reinforcement learning for large language models’, arXiv:2604.09459.

