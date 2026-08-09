# RC-OMD Extended Literature Inventory

## Scope and selection rule

This inventory contains 22 works selected for a specific role in the RC-OMD paper. It is not a general RL bibliography. A work is included only if it supports delayed-reward policy gradients, mirror/KL policy geometry, GRPO/RLVR failure modes, fine-grained credit estimation, uncertainty estimation, or adaptive online optimization.

## Search and verification provenance

- Search date: 9 August 2026.
- A dual-engine OpenAlex/AnySearch hybrid query was attempted for the classical delayed-reward cluster. OpenAlex returned HTTP 429 and the AnySearch-only results were not sufficiently specific, so no entry was accepted solely from that run.
- Metadata and claims were then checked against primary landing pages: arXiv for preprints, PMLR for ICML papers, JMLR for AdaGrad, NeurIPS proceedings for Bootstrapped DQN, the official Sutton/Barto book site, and publisher/DOI records for journal papers.
- Cross-source status: the failed/weak dual-engine response is recorded as a search limitation, not silently treated as corroboration. A fresh novelty search is still required immediately before public submission.

Evidence tags:

- **Foundation**: establishes a definition, estimator, or optimization framework.
- **Direct prior**: operates in the same sparse verifiable reward or token-credit setting.
- **Nearest competitor**: overlaps substantially with the proposed mechanism and must be compared explicitly.
- **Adjacent tool**: supplies a reusable uncertainty or adaptivity idea but is not itself a direct RC-OMD precedent.
- **Survey**: organizes the field; it is not primary evidence for an algorithmic claim.

## A. Classical RL and delayed reward

| Work | Evidence tag | What it establishes | Exact use in RC-OMD | Boundary / caution |
|---|---|---|---|---|
| Sutton and Barto (2018) | Foundation | Standard MDP, return, policy-gradient, and eligibility-trace background | Definitions and classical temporal-credit motivation | Textbook background, not evidence of novelty. |
| Williams (1992) | Foundation | REINFORCE score-function policy gradient | Derivation of terminal-return policy updates | Unbiasedness does not imply low variance or fine-grained credit. |
| Singh and Sutton (1996) | Foundation | Replacing eligibility traces for delayed reinforcement | Classical mechanism that transports later information backward in time | Trace decay is not a reliability estimate and assumes a different learning setup. |
| Schulman *et al.* (2016) | Foundation | GAE bias-variance trade-off through weighted TD residuals | Advantage-estimation baseline and contrast with critic-free RLVR | Requires value estimates; not directly transferable to output-only rewards. |

## B. Mirror descent, KL, and trust-region policy optimization

| Work | Evidence tag | What it establishes | Exact use in RC-OMD | Boundary / caution |
|---|---|---|---|---|
| Beck and Teboulle (2003) | Foundation | Mirror descent with non-Euclidean proximal geometry | Mathematical basis for the OMD update | Convex optimization result, not an RL credit method. |
| Schulman *et al.* (2015) | Foundation | KL-constrained trust-region policy improvement | Why policy drift should be controlled | Global policy trust region, not per-step reliability calibration. |
| Schulman *et al.* (2017) | Foundation | PPO clipped surrogate optimization | Widely used practical proximal baseline | Clipping is not identical to a hard KL guarantee. |
| Geist, Scherrer and Pietquin (2019) | Foundation | General theory of entropy/KL-regularized MDPs | Formal bridge between regularized RL and mirror/proximal optimization | Does not treat sparse verifiable reward credit reliability. |
| Tomar *et al.* (2020) | Direct prior | Practical mirror-descent policy optimization | Closest optimizer-level ancestor | Uses policy-level objectives rather than uncertain step-credit estimates. |
| Duchi, Hazan and Singer (2011) | Adjacent tool | Data-dependent adaptive proximal geometry and regret guarantees | Conceptual basis for adapting local step sizes from observed reliability statistics | Online convex setting; cannot be cited as an RC-OMD guarantee without a new proof. |

## C. Recent RLVR and GRPO

| Work | Evidence tag | What it establishes | Exact use in RC-OMD | Boundary / caution |
|---|---|---|---|---|
| Shao *et al.* (2024) | Direct prior | Introduces GRPO for mathematical reasoning | Defines the group-relative baseline in the target setting | Sequence-level advantage is not a causal credit estimate. |
| DeepSeek-AI (2025) | Direct prior | Large-scale reasoning RL and verifiable outcome rewards in DeepSeek-R1 | Motivation that RLVR is practically important | Scale and performance claims do not validate RC-OMD. |
| Kimi Team (2025) | Direct prior | Long-context RL and practical policy-optimization design for reasoning | Motivation for long-horizon credit and stable updates | Proprietary-scale results are contextual evidence only. |
| He *et al.* (2026a) | Direct prior | Advantage collapse in homogeneous-reward groups and ACR diagnostic | Defines a separate group-level failure mode to monitor | RC-OMD must not claim that reliability scaling creates information in all-correct/all-wrong groups. |
| Mishra, Chakraborty and Kapusuzoglu (2026) | Direct prior | GRPO uniform-credit, gradient-sparsity, and rank-collapse analysis | Strong direct theoretical motivation for non-uniform credit | Diagnosis only; it does not establish our reliability-controlled mirror step. |

## D. Credit assignment and reliability estimation

| Work | Evidence tag | What it establishes | Exact use in RC-OMD | Boundary / caution |
|---|---|---|---|---|
| He *et al.* (2026b) | Nearest competitor | Polarity-entropy analysis and EAPO token weighting | Main entropy-credit baseline | Entropy upper-bounds possible information; it is not ground-truth causal credit or estimator confidence. |
| Xie *et al.* (2026) | Nearest competitor | ACPO with mode-local surrogate entropy | Closest adaptive entropy-based token-credit method | Our distinction must rest on persistent reliability and local mirror geometry, not merely another weight. |
| Ding *et al.* (2026) | Nearest competitor | Counterfactual trajectory comparison for step-sensitive signals | Counterfactual credit baseline/proxy | Additional comparisons may increase rollout cost; 'counterfactual' here is not a blanket causal guarantee. |
| Shan *et al.* (2026) | Nearest competitor | Self-conditioned per-token KL credit from verified trajectories | Self-conditioned credit baseline/proxy | Credit magnitude and reliability remain conceptually separate. |
| Osband *et al.* (2016) | Adjacent tool | Bootstrap-style uncertainty through multiple randomized heads | Justifies testing bootstrap disagreement as a reliability proxy | Exploration/value uncertainty is not the same as step-credit uncertainty. |
| Zhang (2026) | Survey | Taxonomy of reasoning and agentic credit assignment | Coverage check and terminology | Survey is secondary evidence; algorithm claims should cite primary papers above. |
| Zhou *et al.* (2026) | Direct prior | U-statistic interpretation of the GRPO policy gradient | Statistical perspective on grouped estimators | Does not solve step-level credit or validate reliability calibration. |

## Claim-to-citation map for the working paper

| Intended external claim | Primary citations | Do not overstate |
|---|---|---|
| Terminal returns create a temporal-credit and variance problem for policy gradients. | Williams (1992); Sutton and Barto (2018); Schulman *et al.* (2016) | Do not claim every delayed-reward method fails uniformly. |
| KL/proximal control can stabilize policy updates. | Schulman *et al.* (2015); Geist, Scherrer and Pietquin (2019); Tomar *et al.* (2020) | Do not claim a KL bound alone yields correct credit. |
| GRPO uses grouped relative outcomes and can broadcast coarse sequence-level signals. | Shao *et al.* (2024); Mishra, Chakraborty and Kapusuzoglu (2026) | Distinguish implemented GRPO from simplified toy OMD. |
| Homogeneous-reward groups can have vanishing relative advantages. | He *et al.* (2026a) | RC-OMD does not manufacture missing information. |
| Entropy, counterfactual comparison, and self-conditioning are current token/step-credit proxies. | He *et al.* (2026b); Xie *et al.* (2026); Ding *et al.* (2026); Shan *et al.* (2026) | Do not call these proxies causal ground truth. |
| Adaptive geometry and bootstrap disagreement motivate reliability-aware updates. | Duchi, Hazan and Singer (2011); Osband *et al.* (2016) | These are adjacent tools, not direct proofs of RC-OMD. |
| RC-OMD currently has a tabular OOD GO and a shared-linear NO-GO. | This repository's preregistered experiments and paper results | Cite exact repository artifacts and report seeds/configuration; do not use external citations for our result. |

## Reading priority before Cambridge

1. **Must understand in detail:** Williams (1992); Beck and Teboulle (2003); Schulman *et al.* (2015, 2016); Tomar *et al.* (2020); Shao *et al.* (2024); He *et al.* (2026b); Xie *et al.* (2026); Mishra, Chakraborty and Kapusuzoglu (2026).
2. **Must be able to compare experimentally:** He *et al.* (2026a, 2026b); Xie *et al.* (2026); Ding *et al.* (2026); Shan *et al.* (2026).
3. **Background or adjacent method:** Sutton and Barto (2018); Singh and Sutton (1996); Geist, Scherrer and Pietquin (2019); Duchi, Hazan and Singer (2011); Osband *et al.* (2016); Zhang (2026); Zhou *et al.* (2026).

## Bibliography (Harvard style, alphabetical)

Beck, A. and Teboulle, M. (2003) 'Mirror descent and nonlinear projected subgradient methods for convex optimization', *Operations Research Letters*, 31(3), pp. 167-175. Available at: https://doi.org/10.1016/S0167-6377(02)00231-6 (Accessed: 9 August 2026).

DeepSeek-AI (2025) 'DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning', *arXiv preprint*, arXiv:2501.12948. Available at: https://arxiv.org/abs/2501.12948 (Accessed: 9 August 2026).

Ding, F., Zhang, Y., Wang, Y. and Zeng, Z. (2026) 'Reducing credit assignment variance via counterfactual reasoning paths', *arXiv preprint*, arXiv:2605.16302. Available at: https://arxiv.org/abs/2605.16302 (Accessed: 9 August 2026).

Duchi, J., Hazan, E. and Singer, Y. (2011) 'Adaptive subgradient methods for online learning and stochastic optimization', *Journal of Machine Learning Research*, 12(61), pp. 2121-2159. Available at: https://jmlr.org/papers/v12/duchi11a.html (Accessed: 9 August 2026).

Geist, M., Scherrer, B. and Pietquin, O. (2019) 'A theory of regularized Markov decision processes', in Chaudhuri, K. and Salakhutdinov, R. (eds.) *Proceedings of the 36th International Conference on Machine Learning*. PMLR 97, pp. 2160-2169. Available at: https://proceedings.mlr.press/v97/geist19a.html (Accessed: 9 August 2026).

He, X., Sun, Q., Cheng, A., Li, X., Ji, X., Lu, H., Huang, R. and Hu, Q. (2026a) 'Advantage collapse in group relative policy optimization: Diagnosis and mitigation', *arXiv preprint*, arXiv:2605.21125. Available at: https://arxiv.org/abs/2605.21125 (Accessed: 9 August 2026).

He, Y., Wu, H., Liu, S., Ge, H., Zhou, H., Wu, K., Zheng, Z., Lin, Q., Zhong, Z. and Zhang, Y. (2026b) 'Rethinking token-level credit assignment in RLVR: A polarity-entropy analysis', *arXiv preprint*, arXiv:2604.11056. Available at: https://arxiv.org/abs/2604.11056 (Accessed: 9 August 2026).

Kimi Team (2025) 'Kimi k1.5: Scaling reinforcement learning with LLMs', *arXiv preprint*, arXiv:2501.12599. Available at: https://arxiv.org/abs/2501.12599 (Accessed: 9 August 2026).

Mishra, A., Chakraborty, S. and Kapusuzoglu, B. (2026) 'On the policy gradient foundations of group relative policy optimization: Credit assignment, gradient sparsity, and rank collapse', *arXiv preprint*, arXiv:2606.29238. Available at: https://arxiv.org/abs/2606.29238 (Accessed: 9 August 2026).

Osband, I., Blundell, C., Pritzel, A. and Van Roy, B. (2016) 'Deep exploration via bootstrapped DQN', in Lee, D., Sugiyama, M., Luxburg, U., Guyon, I. and Garnett, R. (eds.) *Advances in Neural Information Processing Systems 29*. Available at: https://proceedings.neurips.cc/paper/2016/hash/8d8818c8e140c64c743113f563cf750f-Abstract.html (Accessed: 9 August 2026).

Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015) 'Trust region policy optimization', in Bach, F. and Blei, D. (eds.) *Proceedings of the 32nd International Conference on Machine Learning*. PMLR 37, pp. 1889-1897. Available at: https://proceedings.mlr.press/v37/schulman15.html (Accessed: 9 August 2026).

Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P. (2016) 'High-dimensional continuous control using generalized advantage estimation', *International Conference on Learning Representations*. Available at: https://arxiv.org/abs/1506.02438 (Accessed: 9 August 2026).

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) 'Proximal policy optimization algorithms', *arXiv preprint*, arXiv:1707.06347. Available at: https://arxiv.org/abs/1707.06347 (Accessed: 9 August 2026).

Shan, Y., Guo, Y., Cheng, Z., Liu, Z., Zhu, X., Wang, X., Yao, J., Lin, W., Wang, H. and Huang, H. (2026) 'Learning from own solutions: Self-conditioned credit assignment for reinforcement learning with verifiable rewards', *arXiv preprint*, arXiv:2606.18810. Available at: https://arxiv.org/abs/2606.18810 (Accessed: 9 August 2026).

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y. and Guo, D. (2024) 'DeepSeekMath: Pushing the limits of mathematical reasoning in open language models', *arXiv preprint*, arXiv:2402.03300. Available at: https://arxiv.org/abs/2402.03300 (Accessed: 9 August 2026).

Singh, S.P. and Sutton, R.S. (1996) 'Reinforcement learning with replacing eligibility traces', *Machine Learning*, 22(1-3), pp. 123-158. Available at: https://doi.org/10.1007/BF00114726 (Accessed: 9 August 2026).

Sutton, R.S. and Barto, A.G. (2018) *Reinforcement Learning: An Introduction*. 2nd edn. Cambridge, MA: MIT Press. Available at: http://incompleteideas.net/book/the-book-2nd.html (Accessed: 9 August 2026).

Tomar, M., Shani, L., Efroni, Y. and Ghavamzadeh, M. (2020) 'Mirror Descent Policy Optimization', *arXiv preprint*, arXiv:2005.09814. Available at: https://arxiv.org/abs/2005.09814 (Accessed: 9 August 2026).

Williams, R.J. (1992) 'Simple statistical gradient-following algorithms for connectionist reinforcement learning', *Machine Learning*, 8(3-4), pp. 229-256. Available at: https://doi.org/10.1007/BF00992696 (Accessed: 9 August 2026).

Xie, Z., You, Y., Li, Y., Gong, E., Chen, Z., Chen, Q., Cheng, Y., Jiang, P. and Mu, Y. (2026) 'ACPO: Adaptive credit policy optimization via fine-grained surrogate entropy', *arXiv preprint*, arXiv:2607.03126. Available at: https://arxiv.org/abs/2607.03126 (Accessed: 9 August 2026).

Zhang, C. (2026) 'From reasoning to agentic: Credit assignment in reinforcement learning for large language models', *arXiv preprint*, arXiv:2604.09459. Available at: https://arxiv.org/abs/2604.09459 (Accessed: 9 August 2026).

Zhou, H., Ye, K., Xu, E., Zhu, J., Yang, Y., Gong, S. and Shi, C. (2026) 'Demystifying group relative policy optimization: Its policy gradient is a U-statistic', *arXiv preprint*, arXiv:2603.01162. Available at: https://arxiv.org/abs/2603.01162 (Accessed: 9 August 2026).

## Bibliography maintenance rules

1. Every externally sourced claim in the paper must have an in-text citation at the point of use.
2. Every bibliography entry must be cited in the paper; uncited inventory items stay out of the final bibliography.
3. Final bibliography entries are ordered alphabetically by the first author or corporate author.
4. arXiv entries include authors, year, full title, arXiv identifier, stable URL, and access date.
5. Primary papers support algorithmic claims; surveys support taxonomy and coverage only.
6. New papers are added only when they change the novelty boundary, baseline set, theory, or experimental design.
7. The novelty audit must be repeated immediately before public preprint or conference submission.
