# RC-OMD Workshop — Full Rehearsal and Release Check

## Timing Result: PASS

- Script length: **471 words**.
- Continuous delivery at 115 words per minute: **4:06**.
- Planned pauses and one speaker hand-off: **approximately 30–35 seconds**.
- Expected total: **4:36–4:41**.
- Conservative lower-speed check: at 100 words per minute plus 15 seconds of essential pauses, the talk is **4:58**.
- Speaking plan: HUANG YUXUAN presents Slides 1–3; LI YUFEI presents Slides 4–5. There is only one hand-off.

| Slide | Words | Target delivery | Rehearsal cue |
|---|---:|---:|---|
| 1. Group Introduction | 78 | 0:41 | Introduce roles once; do not read every card twice. |
| 2. Sparse Reward Problem | 89 | 0:46 | Pause after “whole sequence”; emphasize the non-causal boundary. |
| 3. RC-OMD Method | 97 | 0:51 | Point to “where” and “how far”; avoid expanding the estimator derivation. |
| 4. Preregistered OOD GO | 103 | 0:54 | State the frozen rule before the observed numbers; say “Pareto result”. |
| 5. Shared-Parameter NO-GO | 104 | 0:54 | Point to the 0.75 line, then finish with the three professor questions. |

## Numerical Consistency: PASS

- Slide 4 values match `results/ood_preregistered/protocol_evaluation.json` and `docs/MILESTONE_5_OOD_RESULTS.md`.
- OOD decision: **4/4 GO**; displayed AUC differences and KL ratios match the frozen output.
- Slide 5 values match `results/function_approx_preregistered/protocol_evaluation.json` and `docs/MILESTONE_6_FUNCTION_APPROX_RESULTS.md`.
- Shared-parameter decision: **1/3, NO-GO**; all AUC conditions passed and the declared failure came from distractor KL.
- Protocol and execution identifiers are printed in the relevant slide footers.

## Research-Boundary Check: PASS

- The deck explicitly calls the study **associative**, not causal credit assignment.
- The positive result is described as **tabular and factorized**.
- The same-step diagnostic and different preregistered base steps are acknowledged in the script.
- Slide 5 explicitly states: **“NO neural-network or LLM effectiveness claim.”**
- The shared-linear feature result is retained as a **NO-GO**, not softened into a positive transfer claim.

## File and Visual QA: PASS

- Editable PowerPoint imports successfully and contains exactly five slides.
- Every slide contains a `[Sources]` speaker-notes block.
- Expected visible claims and all OOD table values were verified after PPTX round-trip import.
- Layout JSON bounding boxes remain within the 1280 × 720 slide canvas.
- The PDF contains five 16:9 pages and was rendered back to PNG for visual inspection.
- No visible clipping, overlap, or unreadable table cells were observed in the PPTX or PDF renders.
- The repository test suite passes **26/26** tests via `python -m unittest discover -s tests -v`.

## Final Delivery Advice

Do not add extra experimental claims during the live talk. If the professor asks about neural or LLM transfer, answer with the NO-GO evidence first, then present the Fisher-metric and constrained-projection ideas strictly as future algorithmic questions.
