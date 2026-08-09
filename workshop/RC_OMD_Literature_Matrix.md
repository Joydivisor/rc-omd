# RC-OMD One-Page Literature Matrix

This matrix is intentionally limited to the ten works that most directly define the problem, optimizer, nearest competing methods, and remaining research boundary. The broader reading list is maintained in `RC_OMD_Extended_Literature_Inventory.md`.

| Work | Main problem | Signal / granularity | Optimization mechanism | Relevance and remaining boundary |
|---|---|---|---|---|
| Williams (1992), *REINFORCE* | Learning a policy from sampled returns | Trajectory return / action log-probability | Monte Carlo policy gradient | Establishes the unbiased score-function update; terminal returns still give coarse temporal credit. |
| Schulman *et al.* (2016), *GAE* | Bias-variance control in policy-gradient advantage estimates | Temporally weighted TD residuals | Exponentially weighted advantage estimator | Provides the classical credit-estimation baseline; it normally requires a value function and is not a critic-free RLVR method. |
| Beck and Teboulle (2003), *Mirror Descent* | First-order optimization beyond Euclidean geometry | Gradient / decision level | Bregman-proximal mirror step | Supplies the classical OMD geometry; it does not address sparse RL credit. |
| Schulman *et al.* (2015), *TRPO* | Stable policy improvement | Policy-level advantage | KL-constrained trust region | Establishes policy proximity as a stability device; the KL budget is not calibrated by step-credit reliability. |
| Tomar *et al.* (2020), *MDPO* | Practical mirror-descent policy optimization | Policy-level advantage | Divergence-controlled policy update | Connects mirror descent to modern RL; it does not estimate or calibrate step-level credit reliability. |
| Shao *et al.* (2024), *DeepSeekMath* | Critic-free RL for mathematical reasoning | Group-relative sequence advantage | GRPO-style clipped policy optimization | Introduces GRPO in the target setting; a sequence-level outcome is broadcast across response tokens. |
| He *et al.* (2026), *EAPO* | Token-level credit dilution in RLVR | Reward polarity and token entropy | Entropy-aware token weighting | Strong nearest baseline for entropy-based credit; entropy bounds possible credit but is not established as causal credit or confidence. |
| Xie *et al.* (2026), *ACPO* | Fine-grained adaptive token credit | Surrogate entropy / token level | Adaptive credit-weighted policy optimization | Closely related empirical weighting method; it does not study persistent reliability controlling local mirror geometry. |
| Mishra *et al.* (2026), *Policy-Gradient Foundations of GRPO* | Uniform credit, gradient sparsity, and rank collapse | Group-relative sequence signal / token gradients | First-principles GRPO analysis | Formalizes the optimization failure motivating our work; it diagnoses rather than supplies reliability-calibrated OMD. |
| **RC-OMD (this project)** | How aggressively to trust an imperfect credit signal | Persistent online reliability / step level | Reliability-scaled local OMD step or KL budget | OOD GO only in tabular factorized tasks; shared-linear transfer NO-GO under feature aliasing; no causal-credit, neural-network, or LLM claim. |

## Harvard-style references

Beck, A. and Teboulle, M. (2003) 'Mirror descent and nonlinear projected subgradient methods for convex optimization', *Operations Research Letters*, 31(3), pp. 167-175. Available at: https://doi.org/10.1016/S0167-6377(02)00231-6 (Accessed: 9 August 2026).

He, Y., Wu, H., Liu, S., Ge, H., Zhou, H., Wu, K., Zheng, Z., Lin, Q., Zhong, Z. and Zhang, Y. (2026) 'Rethinking token-level credit assignment in RLVR: A polarity-entropy analysis', *arXiv preprint*, arXiv:2604.11056. Available at: https://arxiv.org/abs/2604.11056 (Accessed: 9 August 2026).

Mishra, A., Chakraborty, S. and Kapusuzoglu, B. (2026) 'On the policy gradient foundations of group relative policy optimization: Credit assignment, gradient sparsity, and rank collapse', *arXiv preprint*, arXiv:2606.29238. Available at: https://arxiv.org/abs/2606.29238 (Accessed: 9 August 2026).

Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015) 'Trust region policy optimization', in Bach, F. and Blei, D. (eds.) *Proceedings of the 32nd International Conference on Machine Learning*. PMLR 37, pp. 1889-1897. Available at: https://proceedings.mlr.press/v37/schulman15.html (Accessed: 9 August 2026).

Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P. (2016) 'High-dimensional continuous control using generalized advantage estimation', *International Conference on Learning Representations*. Available at: https://arxiv.org/abs/1506.02438 (Accessed: 9 August 2026).

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y. and Guo, D. (2024) 'DeepSeekMath: Pushing the limits of mathematical reasoning in open language models', *arXiv preprint*, arXiv:2402.03300. Available at: https://arxiv.org/abs/2402.03300 (Accessed: 9 August 2026).

Tomar, M., Shani, L., Efroni, Y. and Ghavamzadeh, M. (2020) 'Mirror Descent Policy Optimization', *arXiv preprint*, arXiv:2005.09814. Available at: https://arxiv.org/abs/2005.09814 (Accessed: 9 August 2026).

Williams, R.J. (1992) 'Simple statistical gradient-following algorithms for connectionist reinforcement learning', *Machine Learning*, 8(3-4), pp. 229-256. Available at: https://doi.org/10.1007/BF00992696 (Accessed: 9 August 2026).

Xie, Z., You, Y., Li, Y., Gong, E., Chen, Z., Chen, Q., Cheng, Y., Jiang, P. and Mu, Y. (2026) 'ACPO: Adaptive credit policy optimization via fine-grained surrogate entropy', *arXiv preprint*, arXiv:2607.03126. Available at: https://arxiv.org/abs/2607.03126 (Accessed: 9 August 2026).
