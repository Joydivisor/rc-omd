# Milestone 2: Corrected Common-Random-Number Reanalysis

Date: 2026-08-14. Erratum reference: [E5](ERRATA.md).

`docs/MILESTONE_2_CREDIT_DIAGNOSTICS.md` is unchanged and remains the record of
what was originally run. This document reports a corrected re-run and the paired
difference between them.

## The defect

`experiments/run_credit_diagnostics.py` seeded each run with:

```python
rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_index, method_index]))
```

Including `method_index` gave each method a different trajectory stream at the
same nominal seed. The comparison was therefore unpaired: part of every reported
method difference was sampling noise between distinct rollout sets rather than a
difference in the methods.

`method_index` is the method's position in the config's method list, so the
numbers also depended on the order methods happened to be written in
`configs/credit_diagnostics.json`.

**Scope.** Milestone 2 only. `experiments/run_reliability_diagnostics.py` seeds
with `np.random.default_rng(seed)` and never mixed in a method index, so
Milestones 3 through 6 always used common random numbers.

## The correction

```python
rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_index]))
```

## Why one method is bitwise unchanged

NumPy treats a trailing zero in a `SeedSequence` entropy list as a no-op:

```text
SeedSequence([7, 1, 0])  ==  SeedSequence([7, 1])      same stream
SeedSequence([7, 1, 1])  !=  SeedSequence([7, 1])      different stream
```

The method at index 0 was therefore already on the stream the correction adopts.
In this config that is `uniform_group_omd`, which reproduces bitwise; the other
two methods move onto its stream.

This is worth stating plainly: the defect silently privileged whichever method
was listed first, and the size of the distortion for every other method depended
on nothing more principled than list order.

## Paired comparison

Ten seeds, `configs/credit_diagnostics.json`.

| Scenario | Method | AUC before | AUC after | Change |
|---|---|---:|---:|---:|
| Entropy aligned | Uniform Group OMD | 0.986363 | 0.986363 | bitwise same |
| Entropy aligned | Entropy-weighted OMD | 0.989172 | 0.989305 | +0.000133 |
| Entropy aligned | Oracle-credit OMD | 0.986409 | 0.986652 | +0.000243 |
| Entropy misleading | Uniform Group OMD | 0.961899 | 0.961899 | bitwise same |
| Entropy misleading | Entropy-weighted OMD | 0.946243 | 0.947095 | +0.000852 |
| Entropy misleading | Oracle-credit OMD | 0.960962 | 0.961780 | +0.000818 |

## Do the Milestone 2 conclusions change?

**No. Both survive, with the same sign.**

| Claim | Gap before | Gap after | Verdict |
|---|---:|---:|---|
| Entropy **helps** when aligned | +0.0028 | +0.0029 | holds |
| Entropy **hurts** when misaligned | -0.0157 | -0.0148 | holds |

The entropy-misleading effect remains roughly five times the entropy-aligned
effect, so the asymmetry noted in [E5](ERRATA.md) is preserved.

## What is still weak

The aligned-scenario effect is now properly paired, but it is still only
**+0.0029 AUC against an across-seed standard deviation of about 0.0004**. No
confidence interval has been computed and no non-inferiority or superiority test
has been pre-declared for Milestone 2, which was a diagnostic milestone rather
than a frozen protocol.

The correct reading is that the direction is real and no longer confounded by
unpaired sampling, not that the magnitude is established. Milestone 2 remains
exploratory evidence, and `paper/CLAIMS.md` continues to list it as such.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_credit_diagnostics --config configs/credit_diagnostics.json
```

The "before" column is from the commit preceding this one; the "after" column is
from this commit. Both were produced on the same machine on 2026-08-14 with
Python 3.12.10 and NumPy 2.5.2.
