"""RWP branch for `qwen-performance-v1`, and its alignment diagnostics.

**The loss lives in `qwen_grpo.py` and is deliberately not duplicated here.**
D5 requires the Uniform baseline to be exactly the reliability-one case of the
RWP objective, and the cheapest way to guarantee that is for one function to
serve both. A second implementation would be a place for the branches to drift
apart in masking, normalization or advantage handling, which is precisely the
failure this protocol cannot tolerate. This module therefore contributes the
branch's reliability configuration and the diagnostics that D2's index-based
alignment demands.

Index alignment is the known weak point of the language-model transfer. D2
records that different rollouts say different things at the same index, so the
diagnostics below exist to quantify how badly that assumption is strained on
real data rather than leaving it as an unmeasured caveat.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from algorithms.qwen_grpo import (
    LossConfig,
    ReliabilityConfig,
    rwp_grpo_loss,
    token_reliability,
)

__all__ = [
    "LossConfig",
    "ReliabilityConfig",
    "alignment_diagnostics",
    "make_rwp_reliability",
    "rwp_grpo_loss",
]


def make_rwp_reliability(**overrides: Any) -> ReliabilityConfig:
    """The RWP branch's reliability configuration; `enabled` is forced on."""

    return ReliabilityConfig(**{**overrides, "enabled": True})


def first_divergence(sequences: torch.Tensor, prompt_length: int,
                     group_size: int) -> list[int]:
    """Index of the first token at which a prompt's rollouts stop agreeing.

    Reported relative to the start of generation. A value equal to the
    generated length means every rollout in that group is identical.
    """

    generated = sequences[:, prompt_length:]
    length = generated.shape[1]
    out: list[int] = []
    for group in generated.view(-1, group_size, length):
        reference = group[0]
        agree = (group == reference[None, :]).all(dim=0)
        mismatch = (~agree).nonzero()
        out.append(int(mismatch[0]) if mismatch.numel() else length)
    return out


def alignment_diagnostics(
    sequences: torch.Tensor,
    prompt_length: int,
    mask: torch.Tensor,
    group_size: int,
    reliability: torch.Tensor,
) -> dict[str, Any]:
    """Every alignment metric Q4B requires, from one rollout batch."""

    generated_mask = mask[:, prompt_length:]
    length = generated_mask.shape[1]
    grouped = generated_mask.view(-1, group_size, length)

    coverage = grouped.sum(dim=1).float()                 # (groups, length)
    full_coverage = (coverage == group_size).float().mean()
    any_coverage = (coverage > 0).float().mean()

    lengths = generated_mask.sum(dim=1).float()
    divergence = first_divergence(sequences, prompt_length, group_size)

    active = generated_mask
    reliability_generated = reliability[:, prompt_length:]
    values = reliability_generated[active]

    # Does reliability drift with position? A strong trend means the estimator
    # is largely reading generation length rather than credit dispersion.
    positions = torch.arange(length, device=mask.device, dtype=torch.float32)
    per_position = torch.where(
        active.any(dim=0),
        (reliability_generated * active).sum(dim=0) / active.sum(dim=0).clamp(min=1),
        torch.full((length,), float("nan"), device=mask.device),
    )
    finite = torch.isfinite(per_position)
    if int(finite.sum()) > 2:
        x = positions[finite]
        y = per_position[finite]
        x = x - x.mean()
        y_centred = y - y.mean()
        denominator = float((x * x).sum() * (y_centred * y_centred).sum()) ** 0.5
        trend = float((x * y_centred).sum() / denominator) if denominator > 0 else 0.0
    else:
        trend = float("nan")

    return OrderedDict([
        ("group_size", int(group_size)),
        ("n_groups", int(grouped.shape[0])),
        ("generated_length", int(length)),
        ("first_divergence_per_group", divergence),
        ("first_divergence_mean", float(sum(divergence) / len(divergence))),
        ("first_divergence_min", int(min(divergence))),
        ("aligned_token_fraction_full_group", float(full_coverage)),
        ("aligned_token_fraction_any", float(any_coverage)),
        ("coverage_per_position_mean", [float(v) for v in coverage.mean(dim=0)]),
        ("completion_length_mean", float(lengths.mean())),
        ("completion_length_min", float(lengths.min())),
        ("completion_length_max", float(lengths.max())),
        ("reliability_mean", float(values.mean()) if values.numel() else float("nan")),
        ("reliability_std", float(values.std(unbiased=False)) if values.numel() > 1
         else 0.0),
        ("reliability_min", float(values.min()) if values.numel() else float("nan")),
        ("reliability_max", float(values.max()) if values.numel() else float("nan")),
        ("reliability_position_trend", trend),
        ("comment",
         "aligned_token_fraction_full_group is the share of (group, index) cells "
         "where every rollout is still generating. It falls as completions "
         "diverge in length, and it is the honest measure of how much of D2's "
         "index alignment is actually supported by data."),
    ])


def equivalence_report(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    config: LossConfig,
) -> dict[str, float]:
    """D5 check: reliability-one RWP must reproduce the baseline exactly."""

    ones = torch.ones_like(mask, dtype=torch.float32)
    rwp, _ = rwp_grpo_loss(logits, sequences, mask, advantages, ones, config)
    from algorithms.qwen_grpo import uniform_grpo_loss

    uniform, _ = uniform_grpo_loss(logits, sequences, mask, advantages, config)
    return {
        "rwp_reliability_one": float(rwp),
        "uniform": float(uniform),
        "absolute_difference": float((rwp - uniform).abs()),
        "identical": bool(torch.equal(rwp, uniform)),
    }
