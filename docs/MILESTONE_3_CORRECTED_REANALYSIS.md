# Milestone 3: Corrected Bootstrap Reanalysis

Date: 2026-08-14. Erratum reference: [E4](ERRATA.md).

`docs/MILESTONE_3_BOOTSTRAP_RC_OMD.md` is unchanged and remains the record of
what was originally run. This document reports a corrected re-run and the paired
difference between them.

## The defect

`credit_estimators/bootstrap.py` measured signal and uncertainty in different
spaces:

- `signal_norm` was the norm of action scores **centered across actions**;
- `uncertainty_norm` was the norm of the **uncentered** bootstrap standard
  deviation.

The exponentiated-gradient update is invariant to a constant shift applied to
all actions at a position. That shared component therefore carries no
information about where the policy will move, but it inflated the uncertainty
term while being excluded from the signal term. Reliability

```text
q_k = max(0, 1 - z * u_k / max(s_k, eps))
```

was consequently biased downward, making every bootstrap variant more
conservative than the update geometry justifies.

## The correction

Bootstrap scores are centered across actions before the standard deviation is
taken, so both norms live in the action-difference subspace the update acts on.

```python
centered_bootstrap_scores = bootstrap_scores - bootstrap_scores.mean(axis=2, keepdims=True)
score_std = centered_bootstrap_scores.std(axis=0, ddof=1)
```

Nothing else changed: same config, same seeds, same environment, same update
rule. Non-bootstrap methods produce bitwise-identical results, which confirms
the change is correctly scoped.

## Paired comparison

Five seeds, `configs/reliability_diagnostics.json`. Only the four bootstrap-based
rows can move; the other rows are reproduced unchanged and are shown for
reference.

| Scenario | Method | AUC before | AUC after | Change | KL frac before | KL frac after |
|---|---|---:|---:|---:|---:|---:|
| Entropy misleading | Uniform Group OMD | 0.9582 | 0.9582 | — | 0.0985 | 0.0985 |
| Entropy misleading | Entropy-weighted | 0.9438 | 0.9438 | — | 0.3206 | 0.3206 |
| Entropy misleading | Global reliability | 0.8917 | **0.9030** | +0.0113 | 0.0758 | 0.0810 |
| Entropy misleading | Local RC-OMD `z=1.96` | 0.8842 | **0.8992** | +0.0150 | 0.0170 | **0.0132** |
| Entropy misleading | Oracle credit | 0.9586 | 0.9586 | — | 0.0000 | 0.0000 |
| Long sparse credit | Uniform Group OMD | 0.9445 | 0.9445 | — | 0.1091 | 0.1091 |
| Long sparse credit | Entropy-weighted | 0.9243 | 0.9243 | — | 0.4809 | 0.4809 |
| Long sparse credit | Global reliability | 0.8594 | **0.8641** | +0.0047 | 0.0828 | 0.0799 |
| Long sparse credit | Local RC-OMD `z=1.96` | 0.8409 | **0.8494** | +0.0085 | 0.0206 | **0.0156** |
| Long sparse credit | Oracle credit | 0.9447 | 0.9447 | — | 0.0000 | 0.0000 |

Threshold ablation, `configs/reliability_ablation.json`:

| Scenario | Variant | AUC before | AUC after | KL frac before | KL frac after |
|---|---|---:|---:|---:|---:|
| Entropy misleading | `z=0.5` | 0.9502 | 0.9512 | 0.0669 | 0.0651 |
| Entropy misleading | `z=1.0` | 0.9349 | 0.9402 | 0.0378 | 0.0267 |
| Entropy misleading | `z=1.0`, floor 0.25 | 0.9423 | 0.9447 | 0.0413 | 0.0355 |
| Entropy misleading | `z=1.96` | 0.8842 | 0.8992 | 0.0170 | 0.0132 |
| Long sparse credit | `z=0.5` | 0.9325 | 0.9337 | 0.0819 | 0.0756 |
| Long sparse credit | `z=1.0` | 0.9117 | 0.9162 | 0.0541 | 0.0458 |
| Long sparse credit | `z=1.0`, floor 0.25 | 0.9206 | 0.9227 | 0.0585 | 0.0519 |
| Long sparse credit | `z=1.96` | 0.8409 | 0.8494 | 0.0206 | 0.0156 |

## Result

**For local RC-OMD the correction is a Pareto improvement, not a trade-off.**
Every local variant gained AUC and simultaneously reduced distractor KL. The
strict `z=1.96` setting gained the most, which is what the mechanism predicts:
the inflation of `u_k` matters most where the confidence multiplier is largest.

Global reliability behaves differently. It gained AUC in both tasks but its
distractor KL fraction rose in the entropy-misleading task, from 0.0758 to
0.0810. Applying a single strongest-position confidence to all positions means a
less conservative estimate also loosens the distractor positions, so the global
ablation does not inherit the local improvement.

## Absolute distractor KL

`docs/MILESTONE_3_BOOTSTRAP_RC_OMD.md` published only the KL *fraction*, which is
the subject of [E2](ERRATA.md). Absolute values are recorded here.

| Scenario | Uniform | RC-OMD before | Reduction | RC-OMD after | Reduction |
|---|---:|---:|---:|---:|---:|
| Entropy misleading | 0.02440 | 0.00247 | 89.9% | 0.00206 | **91.6%** |
| Long sparse credit | 0.02682 | 0.00261 | 90.3% | 0.00216 | **92.0%** |

The 89.9% and 90.3% figures are genuine and were simply never published; the
milestone document printed fractions only. They are recoverable because the
deterministic metrics reproduce exactly.

## Does the Milestone 3 conclusion change?

**No.** The headline finding stands: RC-OMD v1 still does not outperform Uniform
Group OMD overall. Corrected local RC-OMD reaches 0.8992 AUC against Uniform's
0.9582 in the entropy-misleading task, and 0.8494 against 0.9445 in the long
task. Estimator calibration remains the bottleneck, and the oracle-credit
diagnostic still shows that accurate local credit preserves learning speed while
eliminating distractor drift.

What changes is the size of the gap and the reason for it. Part of the
conservatism attributed to the bootstrap estimator was an artifact of measuring
uncertainty outside the update-relevant subspace. The estimator is better than
Milestone 3 reported, and still not good enough.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_reliability_diagnostics --config configs/reliability_diagnostics.json
python -m experiments.run_reliability_diagnostics --config configs/reliability_ablation.json
```

Numbers in the "before" columns are from the commit preceding this one; the
"after" columns are from this commit. Both were produced on the same machine on
2026-08-14 with Python 3.12.10 and NumPy 2.5.2.
