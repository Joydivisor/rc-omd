# Milestone 5: Pre-Registered OOD Validation Protocol

Protocol ID: `ood-v1-2026-08-08`

Status at this commit: **protocol frozen before any OOD result is generated**.

## Purpose

All algorithm development so far used two AND-reward controlled sequence tasks.
This protocol evaluates whether the Online RC-OMD trade-off generalizes to new
reward structures, action cardinalities, horizons, sparsity levels, and group
sizes without changing its reliability hyperparameters.

## Frozen Online RC-OMD parameters

- reliability decay: 0.9;
- confidence multiplier: 1.0;
- warm-up effective samples: 8;
- reliability floor: 0.1.

No value above may be changed after examining OOD results. Base step size remains
an ordinary optimizer parameter. Two comparisons are frozen:

1. **Primary Pareto-matched comparison:** Online RC-OMD at base step 1.25 versus
   Uniform Group OMD at base step 0.75. These values were selected from the
   development sweep before creating the OOD tasks.
2. **Same-step diagnostic:** Online RC-OMD versus Uniform Group OMD, both at base
   step 1.0.

Oracle-credit OMD at step 0.75 is a diagnostic reference, not a claimed upper
bound.

## Previously unseen scenarios

The exact machine-readable definitions are in `configs/ood_preregistered.json`.

1. `threshold_3_of_5_small_group`: reward requires at least three of five pivotal
   decisions; binary actions; group size 32.
2. `threshold_4_of_6_three_actions`: reward requires four of six matches; three
   actions; group size 96.
3. `needle_5_of_5_long_horizon`: all five pivotal decisions must match over a
   fourteen-step trajectory; group size 64.
4. `dense_2_of_6_tiny_group`: only two of six matches are needed; group size 24.

These tasks were not used for method or hyperparameter selection.

## Primary outcomes

For each method and scenario, using ten independent random seeds:

- normalized exact-success AUC;
- cumulative absolute KL on known distractor positions;
- cumulative KL on pivotal positions;
- across-seed standard deviation;
- harmful-update rate;
- CPU runtime per seed.

Generated figures must use absolute distractor KL, not only its fraction. Error
bars denote across-seed SD with the seed as the replication unit (`n=10`).

## Pre-declared Go/No-Go rule

For the primary Pareto-matched pair, a scenario passes if both hold:

1. Online RC-OMD AUC is no more than 0.01 below Uniform Group OMD AUC;
2. Online RC-OMD absolute distractor KL is at most 50% of Uniform Group OMD's.

Generalization receives **Go** if at least three of four scenarios pass. Runtime
is reported separately; Online runtime should be at most 1.5 times Uniform for a
systems-feasibility pass.

The same-step comparison is descriptive and cannot override the primary rule.
No scenario may be removed after results are observed.

## Execution discipline

1. Commit and push this protocol, environment implementation, tests, and config.
2. Record the Git commit used for execution.
3. Run the config once without parameter changes.
4. Preserve all per-seed histories locally under `results/ood_preregistered/`.
5. Report every scenario, including failures.
6. Any later tuning must use a new protocol ID and be labelled post-hoc.

## Command to be run after protocol commit

```powershell
python -m unittest discover -s tests -v
python -m experiments.run_reliability_diagnostics --config configs/ood_preregistered.json
```
