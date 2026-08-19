"""GRPO rollout, reward, advantage and loss for `qwen-performance-v1`.

The loss is written **once**, in the general reliability-weighted form of
`docs/QWEN_PERFORMANCE_V1_DEFINITIONS.md` D4. The Uniform/GRPO baseline is that
same function with `reliability = 1` everywhere, per D5, so the two branches
share one code path and their equivalence is exact rather than approximate.

On-policy assumption: exactly one optimizer step is taken per rollout batch, so
`pi_old` is the detached current policy. This removes a second forward pass and
makes `pi_old` exact rather than a stale snapshot. Taking more than one inner
step would invalidate it, which is why `inner_steps` is not a knob here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


# --- reward -----------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Last number in the completion, commas and currency stripped.

    Deliberately plain, and applied identically to both branches and both data
    tiers, so extraction quality cancels in every paired comparison.
    """

    matches = _NUMBER.findall(text.replace("$", ""))
    if not matches:
        return None
    value = matches[-1].replace(",", "").rstrip(".")
    return value or None


def extract_gold(answer_field: str) -> str | None:
    """GSM8K gold answer, taken from after the '####' marker."""

    if "####" not in answer_field:
        return None
    return extract_answer(answer_field.split("####")[-1])


def numbers_match(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False


def exact_match_reward(completion: str, gold_answer: str) -> float:
    return 1.0 if numbers_match(extract_answer(completion),
                                extract_gold(gold_answer)) else 0.0


# --- advantage --------------------------------------------------------------

def group_relative_advantage(
    rewards: torch.Tensor, group_size: int, epsilon: float = 1e-4
) -> torch.Tensor:
    """GRPO advantage: standardize rewards within each prompt's group.

    A group whose rewards are all identical carries no learning signal, and its
    advantages are exactly zero rather than an amplified rounding error.
    """

    if rewards.ndim != 1 or rewards.shape[0] % group_size != 0:
        raise ValueError("rewards must be 1-D with length divisible by group_size")
    grouped = rewards.view(-1, group_size)
    centred = grouped - grouped.mean(dim=1, keepdim=True)
    spread = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantage = torch.where(spread > epsilon, centred / (spread + epsilon),
                            torch.zeros_like(centred))
    return advantage.reshape(-1)


# --- masking ----------------------------------------------------------------

def completion_mask(
    sequences: torch.Tensor, prompt_length: int, pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """True exactly on generated, non-padding tokens up to and including EOS.

    Prompt tokens are excluded because no action was taken at them. Padding is
    excluded so it cannot contribute to the loss. The first EOS is kept -- it is
    a real decision -- and everything after it is dropped.
    """

    batch, total = sequences.shape
    mask = torch.zeros((batch, total), dtype=torch.bool, device=sequences.device)
    mask[:, prompt_length:] = True
    mask &= sequences != pad_token_id

    generated = sequences[:, prompt_length:]
    is_eos = generated == eos_token_id
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(
        has_eos, is_eos.float().argmax(dim=1),
        torch.full((batch,), generated.shape[1] - 1, device=sequences.device),
    )
    positions = torch.arange(generated.shape[1], device=sequences.device)
    keep = positions[None, :] <= first_eos[:, None]
    mask[:, prompt_length:] &= keep
    return mask


# --- reliability ------------------------------------------------------------

@dataclass
class ReliabilityConfig:
    """D2 parameters. Not inherited from the synthetic runs; selected at Q4C."""

    warmup: float = 8.0
    confidence: float = 1.0
    floor: float = 0.1
    enabled: bool = False  # False => r = 1 everywhere => Uniform baseline


def token_reliability(
    advantages: torch.Tensor, mask: torch.Tensor, group_size: int,
    prompt_length: int, config: ReliabilityConfig,
) -> torch.Tensor:
    """Per-token reliability from the rollout group, by D2.

    Tokens are aligned across rollouts by index. This is crude -- different
    rollouts say different things at the same index -- and is adopted knowingly;
    see the limitation recorded in D2.
    """

    batch, total = mask.shape
    if not config.enabled:
        return torch.ones((batch, total), dtype=torch.float32, device=mask.device)

    generated_mask = mask[:, prompt_length:].view(-1, group_size,
                                                  total - prompt_length)
    grouped_advantage = advantages.view(-1, group_size)
    n_groups, _, length = generated_mask.shape

    coverage = generated_mask.sum(dim=1).float()                    # (groups, len)
    dispersion = torch.zeros((n_groups, length), device=mask.device)
    for position in range(length):
        alive = generated_mask[:, :, position]
        for g in range(n_groups):
            selected = grouped_advantage[g][alive[g]]
            if selected.numel() > 1:
                dispersion[g, position] = selected.std(unbiased=False)

    raw = (coverage / (coverage + config.warmup)) / (1.0 + config.confidence * dispersion)
    raw = raw.clamp(min=config.floor)

    reliability = torch.ones((batch, total), dtype=torch.float32, device=mask.device)
    reliability[:, prompt_length:] = raw.repeat_interleave(group_size, dim=0)
    return reliability


def mean_one(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """D3: normalize over the generated tokens of the optimization batch."""

    active = mask.sum()
    if active == 0:
        return weights
    mean = (weights * mask).sum() / active
    return weights / mean.clamp(min=1e-8)


# --- loss -------------------------------------------------------------------

@dataclass
class LossConfig:
    eta: float = 1.0
    lam: float = 1.0
    chunk_tokens: int = 512
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)


def rwp_grpo_loss(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    reliability: torch.Tensor,
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The D4 loss. `reliability = 1` recovers the Uniform/GRPO baseline exactly.

    ``logits`` are the trainable policy's, shape (batch, total, vocab), already
    shifted so that ``logits[:, t]`` predicts ``sequences[:, t + 1]``. ``pi_old``
    is the detached softmax of the same logits, which is exact under the
    one-inner-step assumption documented at module level.
    """

    batch, length, _ = logits.shape
    targets = sequences[:, 1:]
    active = mask[:, 1:]
    advantage_per_token = advantages[:, None].expand(-1, length - 0)[:, : targets.shape[1]]
    reliability_token = reliability[:, 1:]

    weights = reliability_token + config.lam * (1.0 - reliability_token)
    weights = mean_one(weights, active)

    flat_logits = logits[:, :-1]
    total_loss = flat_logits.new_zeros(())
    n_active = active.sum().clamp(min=1)

    diagnostics_kl = flat_logits.new_zeros(())
    for start in range(0, targets.shape[1], config.chunk_tokens):
        stop = min(start + config.chunk_tokens, targets.shape[1])
        chunk_logits = flat_logits[:, start:stop]
        chunk_targets = targets[:, start:stop]
        chunk_active = active[:, start:stop]
        if not chunk_active.any():
            continue
        chunk_advantage = advantage_per_token[:, start:stop]
        chunk_r = reliability_token[:, start:stop]
        chunk_w = weights[:, start:stop]

        log_pi = F.log_softmax(chunk_logits.float(), dim=-1)
        pi_old = log_pi.detach().exp()

        sampled_old = pi_old.gather(-1, chunk_targets[..., None]).squeeze(-1)
        sampled_log = log_pi.gather(-1, chunk_targets[..., None]).squeeze(-1)

        exp_eta_a = torch.exp(config.eta * chunk_advantage).clamp(max=1e4)
        partition = 1.0 + sampled_old * (exp_eta_a - 1.0)
        w_scalar = chunk_r + config.lam * (1.0 - chunk_r)
        c1 = (chunk_r / partition + config.lam * (1.0 - chunk_r)) / w_scalar
        c2 = (chunk_r * exp_eta_a / partition + config.lam * (1.0 - chunk_r)) / w_scalar

        full_ce = -(pi_old * log_pi).sum(dim=-1)
        term = c1 * full_ce - (c2 - c1) * sampled_old * sampled_log
        total_loss = total_loss + (term * chunk_w * chunk_active).sum()
        diagnostics_kl = diagnostics_kl + (full_ce * chunk_active).sum().detach()

    loss = total_loss / n_active
    stats = {
        "active_tokens": float(n_active),
        "mean_reliability": float((reliability_token * active).sum() / n_active),
        "mean_weight": float((weights * active).sum() / n_active),
        "mean_pi_old_entropy": float(diagnostics_kl / n_active),
    }
    return loss, stats


def uniform_grpo_loss(
    logits: torch.Tensor, sequences: torch.Tensor, mask: torch.Tensor,
    advantages: torch.Tensor, config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """D5: the baseline is the RWP loss with reliability identically one."""

    ones = torch.ones_like(mask, dtype=torch.float32)
    return rwp_grpo_loss(logits, sequences, mask, advantages, ones, config)
