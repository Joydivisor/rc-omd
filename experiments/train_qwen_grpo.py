"""GRPO training loop for `qwen-performance-v1`, both branches.

One loop serves Uniform and RWP. The only difference is whether reliability is
computed from the rollout group or held at one, per D5, so the branches cannot
drift apart in the sampling, reward, advantage, masking, optimizer or schedule.

`--verify` runs the Q4A integration gates on a handful of steps and writes a
report instead of training: identical seeds reproduce completions, rewards and
advantages; base weights are untouched; and a checkpoint round trip reproduces
the next step. Those need the real model, so they cannot live in the CPU unit
tests.

Resource discipline matches the smoke test: the process caps its own CUDA
allocation, microbatches the backward pass, and aborts cleanly on OOM or a
non-finite loss.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from algorithms.qwen_grpo import (
    LossConfig,
    ReliabilityConfig,
    completion_mask,
    exact_match_reward,
    group_relative_advantage,
    rwp_grpo_loss,
    token_reliability,
)

REPO = Path(__file__).resolve().parent.parent
_OOM = tuple({torch.OutOfMemoryError, torch.cuda.OutOfMemoryError})


def base_weight_digest(model) -> str:
    """Fingerprint of every non-LoRA parameter, to prove they never move."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if "lora_" in name:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_branch(config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"],
        dtype=getattr(torch, spec["dtype"]),
        attn_implementation=spec.get("attn_implementation", "sdpa"),
    ).to("cuda")
    model = get_peft_model(model, LoraConfig(**config["lora"]))
    # Gradient checkpointing is not optional at 8 GB: without it every layer's
    # activations for the full sequence stay resident through the backward.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return tokenizer, model


def rollout(model, tokenizer, prompts, generation, seed: int):
    """Generate `G` completions per prompt under a seed fixed for this step."""

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=int(generation["max_prompt_tokens"])).to("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    with torch.no_grad():
        sequences = model.generate(
            **batch,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=True,
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            num_return_sequences=int(generation["completions_per_prompt"]),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    prompt_length = batch["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        sequences[:, prompt_length:], skip_special_tokens=True
    )
    return sequences, prompt_length, completions


def score(completions, prompts, golds, group_size):
    """Rewards in rollout order, then group-relative advantages."""

    rewards = []
    for index, completion in enumerate(completions):
        rewards.append(exact_match_reward(completion, golds[index // group_size]))
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
    return reward_tensor, group_relative_advantage(reward_tensor, group_size)


def step_loss(model, sequences, prompt_length, mask, advantages, reliability,
              loss_config, microbatch: int):
    """Accumulate the loss over microbatches, weighted by active-token share."""

    model.train()
    offset_all = sequences.shape[1] - (sequences.shape[1] - prompt_length + 1)
    total_active = mask[:, offset_all + 1:].sum()
    reported = 0.0
    stats: dict[str, float] = {}
    for start in range(0, sequences.shape[0], microbatch):
        stop = min(start + microbatch, sequences.shape[0])
        sub_mask = mask[start:stop]
        if not sub_mask[:, 1:].any():
            continue
        # Only the generated region needs logits. Computing the LM head over
        # the prompt as well would materialise a (prompt x 151936) tensor that
        # the mask discards anyway -- on these lengths that is most of it.
        keep = sequences.shape[1] - prompt_length + 1
        out = model(input_ids=sequences[start:stop], logits_to_keep=keep)
        offset = sequences.shape[1] - keep
        loss, stats = rwp_grpo_loss(
            out.logits, sequences[start:stop, offset:], sub_mask[:, offset:],
            advantages[start:stop], reliability[start:stop, offset:], loss_config,
        )
        share = sub_mask[:, offset + 1:].sum() / total_active
        (loss * share).backward()
        reported += float(loss.detach()) * float(share)
        del out, loss
    return reported, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--branch", choices=["uniform", "rwp"], required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    guard = config["resource_guard"]
    torch.cuda.set_per_process_memory_fraction(
        float(guard["cuda_memory_fraction"]), 0
    )

    tokenizer, model = load_branch(config)
    reliability_config = ReliabilityConfig(
        **{**config["reliability"], "enabled": arguments.branch == "rwp"}
    )
    loss_config = LossConfig(
        eta=float(config["training"]["eta"]),
        lam=float(config["training"]["lam"]),
        chunk_tokens=int(config["training"]["chunk_tokens"]),
        reliability=reliability_config,
    )
    group_size = int(config["generation"]["completions_per_prompt"])
    prompts = list(config["prompts"])
    golds = list(config["gold_answers"])
    seed = int(config["training"]["seed"])

    report: dict[str, Any] = OrderedDict()
    report["branch"] = arguments.branch
    report["protocol_id"] = config["protocol_id"]
    report["reliability_enabled"] = reliability_config.enabled

    digest_before = base_weight_digest(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )

    history = []
    try:
        for step in range(int(config["training"]["steps"])):
            step_seed = seed + step
            sequences, prompt_length, completions = rollout(
                model, tokenizer, prompts, config["generation"], step_seed
            )
            rewards, advantages = score(completions, prompts, golds, group_size)
            mask = completion_mask(sequences, prompt_length,
                                   tokenizer.pad_token_id, tokenizer.eos_token_id)
            reliability = token_reliability(
                rewards, sequences, mask, group_size, prompt_length,
                reliability_config,
            )
            started = time.perf_counter()
            loss_value, stats = step_loss(
                model, sequences, prompt_length, mask, advantages, reliability,
                loss_config, int(config["training"]["microbatch_sequences"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            history.append(OrderedDict([
                ("step", step),
                ("seed", step_seed),
                ("loss", loss_value),
                ("mean_reward", float(rewards.mean())),
                ("reward_sum", float(rewards.sum())),
                ("advantage_abs_mean", float(advantages.abs().mean())),
                ("active_tokens", stats.get("active_tokens")),
                ("mean_reliability", stats.get("mean_reliability")),
                ("seconds", time.perf_counter() - started),
                ("peak_mib", torch.cuda.max_memory_allocated() / 1024**2),
            ]))
            if arguments.verify and step == 0:
                report["verify_first_step"] = OrderedDict([
                    ("completions_head", completions[0][:200]),
                    ("rewards", [float(x) for x in rewards]),
                    ("advantages", [float(x) for x in advantages]),
                    ("active_tokens", stats.get("active_tokens")),
                ])
    except _OOM as error:
        report["failure"] = f"OOM: {error}"
        report["verdict"] = "HALT_OOM"
        _write(arguments, config, report)
        raise SystemExit(1)

    report["history"] = history
    report["base_weights_unchanged"] = base_weight_digest(model) == digest_before
    report["base_weight_digest"] = digest_before

    checkpoint = Path(config["output_directory"]) / f"{arguments.branch}_adapter"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))
    report["checkpoint"] = str(checkpoint)

    if arguments.verify:
        pre, payload = _verify_pre(config, model, tokenizer, prompts, golds,
                                   group_size, seed, reliability_config,
                                   loss_config)
        # The trained model must be released HERE, in the scope that owns it.
        # Releasing inside the helper leaves this reference alive, so the
        # restore copy would load alongside it: ~6.2 GB of weights on an 8 GB
        # card. That mistake cost two runs before it was pinned down.
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        report["verify"] = {**pre, **_verify_post(config, checkpoint, payload,
                                                  loss_config)}

    if arguments.verify:
        v = report["verify"]
        difference = abs(v["loss_before_checkpoint"] - v["loss_after_restore"])
        v["checkpoint_difference"] = difference
        v["checkpoint_consistent"] = difference <= 1e-4
        gates = [report.get("base_weights_unchanged", False),
                 v["same_seed_same_completions"], v["same_seed_same_rewards"],
                 v["same_seed_same_advantages"], v["checkpoint_consistent"]]
        report["verdict"] = "PASS" if all(gates) else "HALT_GATE_FAILED"
        report["failed_gates"] = [
            name for name, ok in zip(
                ["base_weights_unchanged", "same_seed_same_completions",
                 "same_seed_same_rewards", "same_seed_same_advantages",
                 "checkpoint_consistent"], gates) if not ok
        ]
    else:
        report["verdict"] = ("PASS" if report.get("base_weights_unchanged", False)
                             else "HALT")
    _write(arguments, config, report)
    print(f"branch={arguments.branch} verdict={report['verdict']} "
          f"steps={len(history)} peak={history[-1]['peak_mib']:.0f} MiB")


def _verify_pre(config, model, tokenizer, prompts, golds, group_size,
                seed, reliability_config, loss_config):
    """Determinism gates and the pre-checkpoint loss. Holds the trained model."""

    out: dict[str, Any] = OrderedDict()

    a = rollout(model, tokenizer, prompts, config["generation"], seed + 999)
    b = rollout(model, tokenizer, prompts, config["generation"], seed + 999)
    out["same_seed_same_completions"] = bool(torch.equal(a[0], b[0]))
    ra, adv_a = score(a[2], prompts, golds, group_size)
    rb, adv_b = score(b[2], prompts, golds, group_size)
    out["same_seed_same_rewards"] = bool(torch.equal(ra, rb))
    out["same_seed_same_advantages"] = bool(torch.equal(adv_a, adv_b))

    c = rollout(model, tokenizer, prompts, config["generation"], seed + 1000)
    out["different_seed_differs"] = not bool(
        a[0].shape == c[0].shape and torch.equal(a[0], c[0])
    )

    sequences, prompt_length, _ = a
    mask = completion_mask(sequences, prompt_length, tokenizer.pad_token_id,
                           tokenizer.eos_token_id)
    reliability = token_reliability(ra, sequences, mask, group_size,
                                    prompt_length, reliability_config)
    keep = sequences.shape[1] - prompt_length + 1
    offset = sequences.shape[1] - keep
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=sequences[:1], logits_to_keep=keep).logits
        before, _ = rwp_grpo_loss(logits, sequences[:1, offset:],
                                  mask[:1, offset:], adv_a[:1],
                                  reliability[:1, offset:], loss_config)
    out["loss_before_checkpoint"] = float(before)
    payload = OrderedDict([
        ("sequences", sequences[:1].detach().cpu()),
        ("mask", mask[:1].detach().cpu()),
        ("advantages", adv_a[:1].detach().cpu()),
        ("reliability", reliability[:1].detach().cpu()),
        ("keep", int(keep)), ("offset", int(offset)),
    ])
    del logits
    return out, payload


def _verify_post(config, checkpoint, payload, loss_config):
    """Reload from the checkpoint and reproduce the loss. Model released first."""

    out: dict[str, Any] = OrderedDict()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    spec = config["model"]
    base = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], dtype=getattr(torch, spec["dtype"]),
        attn_implementation=spec.get("attn_implementation", "sdpa"),
    ).to("cuda")
    restored = PeftModel.from_pretrained(base, str(checkpoint)).eval()
    sequences = payload["sequences"].to("cuda")
    with torch.no_grad():
        logits = restored(input_ids=sequences,
                          logits_to_keep=payload["keep"]).logits
        after, _ = rwp_grpo_loss(
            logits, sequences[:, payload["offset"]:],
            payload["mask"].to("cuda")[:, payload["offset"]:],
            payload["advantages"].to("cuda"),
            payload["reliability"].to("cuda")[:, payload["offset"]:],
            loss_config,
        )
    out["loss_after_restore"] = float(after)
    return out


def _write(arguments, config, report) -> None:
    target = arguments.output or (
        Path(config["output_directory"]) / f"{arguments.branch}_report.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
