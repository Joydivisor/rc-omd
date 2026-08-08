# Claims and evidence boundary

## Supported by the frozen OOD protocol

- Online RC-OMD passed all four pre-declared controlled scenarios.
- Its normalized success AUC was within 0.0018 of the selected Uniform Group OMD
  comparison in every scenario.
- It used 18.9%--24.8% as much absolute cumulative distractor-position KL.
- Its measured CPU runtime ratio was 1.069--1.107.

## Supported only by exploratory development experiments

- Entropy can help when aligned with positional importance and hurt when the
  alignment is reversed.
- Online moments are much cheaper than the implemented per-group bootstrap
  estimator.
- Reliability decay changes a ranking-lag trade-off.

## Not yet supported

- Causal identification of step credit.
- Improved reward at a fixed nominal step size in every task.
- Robustness under persistent confounding or nonstationarity.
- Benefits under shared-parameter function approximation.
- Benefits for language-model RLVR.

Any abstract, presentation, or submission must preserve these distinctions.
