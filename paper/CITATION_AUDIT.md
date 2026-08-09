# RC-OMD Citation Audit

Audit date: 9 August 2026

## Current working-paper status

- Unique in-text citation keys: 6
- Unique bibliography entries: 6
- In-text citations missing from bibliography: 0
- Bibliography entries not cited in the text: 0

Current closed set:

- `beck2003mirror`
- `he2026eapo`
- `shao2024deepseekmath`
- `tomar2020mdpo`
- `xie2026acpo`
- `zhang2026survey`

The current bibliography is internally consistent but not yet broad enough for a formal submission. The extended inventory in `workshop/RC_OMD_Extended_Literature_Inventory.md` is a candidate pool, not a command to add every item.

## Required additions by paper section

| Paper section / claim | Candidate primary references | Status |
|---|---|---|
| Delayed terminal reward and policy-gradient variance | Williams (1992); Schulman *et al.* (2016) | Add when the classical problem statement is expanded. |
| Eligibility traces as classical temporal credit | Singh and Sutton (1996); Sutton and Barto (2018) | Add only if traces are discussed explicitly. |
| KL trust regions and proximal policy control | Schulman *et al.* (2015, 2017); Geist, Scherrer and Pietquin (2019) | Add to optimizer background. |
| Adaptive online geometry | Duchi, Hazan and Singer (2011) | Add when motivating reliability-dependent geometry; do not imply its regret bound transfers automatically. |
| Modern reasoning RL context | DeepSeek-AI (2025); Kimi Team (2025) | Add only to the motivation/related-work context. |
| GRPO collapse and estimator structure | He *et al.* (2026a); Mishra, Chakraborty and Kapusuzoglu (2026); Zhou *et al.* (2026) | Add to failure-mode and statistical analysis. |
| Alternative fine-grained credit proxies | Ding *et al.* (2026); Shan *et al.* (2026) | Add to nearest-method comparison. |
| Bootstrap uncertainty | Osband *et al.* (2016) | Add only if bootstrap disagreement is actually implemented or analyzed. |

## Release gate

Before any arXiv or workshop submission:

1. Re-run a citation-key closure check.
2. Remove every bibliography entry not cited in the text.
3. Ensure each major external claim points to a primary source where possible.
4. Keep surveys for taxonomy and coverage, not as sole evidence for algorithmic claims.
5. Repeat the recent-work novelty search and update access dates if the bibliography is regenerated.
