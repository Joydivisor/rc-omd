"""Paired Uniform/RWP GRPO on threshold-screened MATH prompts, scored on MATH-500.

The change from the first attempt is the training pool. Prompts are screened so
the base model solves each one sometimes but not always, which is the only
regime where GRPO's group-relative advantage is non-zero. The GSM8K run had 12%
of groups carrying gradient and a level-4/5 MATH pool still had only 19%,
because hard problems mostly add all-wrong groups, which are as gradient-free as
all-right ones. This pool targets the model's decision boundary instead.

Screening uses the base model that both branches share, and the resulting pool
and its order are identical for every branch and seed, so it cannot advantage
either arm -- it only stops both from training on nothing.

Training and evaluation are both MATH, so this is an in-distribution comparison
rather than the transfer test the first attempt ran.

Writes the Section 12 artifacts for every run. Resource discipline matches the
rest of the phase: the process caps its own CUDA allocation, microbatches the
backward pass, and aborts cleanly rather than retrying into a wall.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
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
    group_relative_advantage,
    rwp_grpo_loss,
    token_reliability,
)
from experiments.math500_eval import extract_boxed, score_completion
from experiments.qwen_data import partition  # noqa: F401  (kept for provenance)
from experiments.qwen_q3 import load, manifest

_OOM = tuple({torch.OutOfMemoryError, torch.cuda.OutOfMemoryError})


def gpu_temperature() -> int | None:
    """Current GPU temperature, or None if nvidia-smi is unavailable."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:
        return None


def thermal_gate(config: dict, label: str = "") -> float:
    """Block until the GPU has cooled below the resume threshold.

    The GPU's power limit is firmware-locked at 115 W here and nvidia-smi
    refuses to lower it, so waiting is the only way to hold the machine off its
    thermal ceiling. GPU temperature is used as the proxy for chassis heat
    because Windows does not expose a CPU sensor on this hardware, and the CPU
    reading was heat-soak driven rather than load driven.

    Never blocks indefinitely: if the sensor is unreadable or the ceiling is not
    cleared within the cap, it returns and lets the run proceed rather than
    hanging silently.
    """

    pause_at = float(config.get("thermal_pause_c", 78))
    resume_at = float(config.get("thermal_resume_c", 65))
    cap = float(config.get("thermal_max_wait_s", 900))

    temperature = gpu_temperature()
    if temperature is None or temperature < pause_at:
        return 0.0

    waited = 0.0
    print(f"  [thermal] {temperature}C >= {pause_at:.0f}C, pausing {label}",
          flush=True)
    while temperature is not None and temperature > resume_at and waited < cap:
        time.sleep(10.0)
        waited += 10.0
        temperature = gpu_temperature()
    print(f"  [thermal] resumed at {temperature}C after {waited:.0f}s", flush=True)
    return waited


def cooldown(worked_seconds: float, factor: float) -> float:
    """Idle for a fraction of the work just done, to cap average power.

    The GPU's power limit is firmware-locked at 115 W on this machine and
    nvidia-smi cannot lower it, so duty cycling is the only lever available for
    holding average draw down. Sleeping proportionally to the work keeps the
    ratio stable whether a step was fast or slow.
    """

    if factor <= 0.0:
        return 0.0
    pause = worked_seconds * factor
    time.sleep(pause)
    return pause

MATH_PROMPT = (
    "Solve the mathematics problem.\n"
    "Show concise reasoning and put your final answer in \\boxed{}."
)


def training_prompts(pool_path, needed: int):
    """Threshold-screened MATH prompts, in the pool's frozen order.

    The order is NOT seed-shuffled: every branch and seed must see the same
    prompts in the same sequence, so the only thing a seed changes is rollout
    sampling.
    """

    pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))["prompts"]
    return pool[:needed]


def math500_subset(count: int, levels):
    from datasets import load_dataset

    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"problem": p, "answer": a, "unique_id": u, "level": int(l)}
            for p, a, u, l in zip(data["problem"], data["answer"],
                                  data["unique_id"], data["level"])
            if int(l) in set(levels)]
    # Deterministic and content-addressed: sorting by unique_id fixes the subset
    # before any model is run and makes it identical on every machine.
    rows.sort(key=lambda r: r["unique_id"])
    return rows[:count]


def evaluate_math500(model, tokenizer, rows, generation,
                     cooldown_factor: float = 0.0,
                     thermal_config: dict | None = None) -> dict[str, Any]:
    thermal_config = thermal_config or {}
    model.eval()
    was_cache = model.config.use_cache
    model.config.use_cache = True
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{MATH_PROMPT}\n\n{r['problem']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]
    batch_size = int(generation["eval_batch"])
    correct: list[float] = []
    boxed_found = 0
    started = time.perf_counter()
    thermal_total = 0.0
    for start in range(0, len(rows), batch_size):
        thermal_total += thermal_gate(thermal_config, "before eval batch")
        batch_started = time.perf_counter()
        chunk = texts[start:start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=768).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=int(generation["eval_max_new_tokens"]),
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completions = tokenizer.batch_decode(
            out[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for offset, completion in enumerate(completions):
            row = rows[start + offset]
            correct.append(score_completion(completion, row["answer"]))
            boxed_found += int(extract_boxed(completion) is not None)
        del out, encoded
        cooldown(time.perf_counter() - batch_started, cooldown_factor)
    model.config.use_cache = was_cache
    return {
        "n": len(rows),
        "thermal_wait_seconds": thermal_total,
        "accuracy": sum(correct) / len(correct),
        "boxed_rate": boxed_found / len(rows),
        "per_item_correct": correct,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--branch", choices=["uniform", "rwp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluate the untrained checkpoint as the reference")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    torch.cuda.set_per_process_memory_fraction(
        float(config["resource_guard"]["cuda_memory_fraction"]), 0
    )
    tag = "reference" if arguments.eval_only else f"{arguments.branch}_s{arguments.seed}"
    out_dir = Path(config["output_directory"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load(config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    record: dict[str, Any] = OrderedDict()
    record["tag"] = tag
    record["branch"] = arguments.branch
    record["seed"] = arguments.seed
    record["eval_only"] = arguments.eval_only
    record["training_data"] = ("EleutherAI/hendrycks_math train Levels 3-5, "
                               "threshold-screened to non-degenerate groups")
    record["evaluation_data"] = ("HuggingFaceH4/MATH-500 Levels 4-5, frozen "
                                 "subset by unique_id")
    record["note"] = ("In-distribution: MATH train -> MATH-500. Training pool "
                      "is screened so the base model is at threshold on every "
                      "prompt; screening is shared by both branches.")

    training = config["training"]
    generation = config["generation"]
    group_size = int(generation["completions_per_prompt"])
    metrics_path = out_dir / "metrics.jsonl"

    if not arguments.eval_only:
        reliability_config = ReliabilityConfig(
            **{**config["reliability"], "enabled": arguments.branch == "rwp"}
        )
        loss_config = LossConfig(
            eta=float(training["eta"]), lam=float(training["lam"]),
            chunk_tokens=int(training["chunk_tokens"]),
            reliability=reliability_config,
        )
        steps = int(training["steps"])
        per_step = int(training["prompts_per_step"])
        prompts = training_prompts(arguments.pool, steps * per_step)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(training["learning_rate"]),
        )
        history = []
        with metrics_path.open("w", encoding="utf-8") as handle:
            try:
                for step in range(steps):
                    thermal_waited = thermal_gate(config, f"before step {step}")
                    chunk = prompts[step * per_step:(step + 1) * per_step]
                    texts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user",
                              "content": MATH_PROMPT + '\n\n' + r["problem"]}],
                            tokenize=False, add_generation_prompt=True)
                        for r in chunk
                    ]
                    encoded = tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=int(generation["max_prompt_tokens"]),
                    ).to("cuda")
                    step_seed = arguments.seed * 1000 + step
                    torch.manual_seed(step_seed)
                    torch.cuda.manual_seed_all(step_seed)
                    model.eval()
                    model.config.use_cache = True
                    started = time.perf_counter()
                    with torch.no_grad():
                        sequences = model.generate(
                            **encoded,
                            max_new_tokens=int(generation["max_new_tokens"]),
                            do_sample=True,
                            temperature=float(generation["temperature"]),
                            top_p=float(generation["top_p"]),
                            num_return_sequences=group_size,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True,
                        )
                    model.config.use_cache = False
                    prompt_length = encoded["input_ids"].shape[1]
                    completions = tokenizer.batch_decode(
                        sequences[:, prompt_length:], skip_special_tokens=True
                    )
                    rewards = torch.tensor(
                        [score_completion(c, chunk[i // group_size]["gold"])
                         for i, c in enumerate(completions)],
                        dtype=torch.float32, device="cuda",
                    )
                    advantages = group_relative_advantage(rewards, group_size)
                    mask = completion_mask(sequences, prompt_length,
                                           tokenizer.pad_token_id,
                                           tokenizer.eos_token_id)
                    reliability = token_reliability(
                        rewards, sequences, mask, group_size, prompt_length,
                        reliability_config,
                    )

                    model.train()
                    keep = sequences.shape[1] - prompt_length + 1
                    offset = sequences.shape[1] - keep
                    total_active = mask[:, offset + 1:].sum()
                    loss_value = 0.0
                    grad_norm = 0.0
                    for begin in range(0, sequences.shape[0],
                                       int(training["microbatch_sequences"])):
                        stop = begin + int(training["microbatch_sequences"])
                        sub_mask = mask[begin:stop]
                        if not sub_mask[:, offset + 1:].any():
                            continue
                        result = model(input_ids=sequences[begin:stop],
                                       logits_to_keep=keep)
                        loss, _ = rwp_grpo_loss(
                            result.logits, sequences[begin:stop, offset:],
                            sub_mask[:, offset:], advantages[begin:stop],
                            reliability[begin:stop, offset:], loss_config,
                        )
                        share = sub_mask[:, offset + 1:].sum() / total_active
                        (loss * share).backward()
                        loss_value += float(loss.detach()) * float(share)
                        del result, loss
                    grad_norm = float(torch.sqrt(sum(
                        (p.grad.float() ** 2).sum() for p in model.parameters()
                        if p.requires_grad and p.grad is not None
                    )))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()

                    row = OrderedDict([
                        ("step", step), ("seed", step_seed),
                        ("loss", loss_value), ("grad_norm", grad_norm),
                        ("mean_reward", float(rewards.mean())),
                        ("reward_variance_groups", float(
                            sum(1 for g in rewards.view(-1, group_size)
                                if len(set(g.tolist())) > 1) / (len(rewards) // group_size)
                        )),
                        ("mean_reliability", float(
                            (reliability[:, offset:][:, 1:] * mask[:, offset + 1:]).sum()
                            / total_active.clamp(min=1))),
                        ("active_tokens", int(total_active)),
                        ("mean_generation_length", float(
                            mask[:, prompt_length:].sum(dim=1).float().mean())),
                        ("seconds", time.perf_counter() - started),
                        ("peak_mib", torch.cuda.max_memory_allocated() / 1024**2),
                    ])
                    row["thermal_wait_seconds"] = thermal_waited
                    row["cooldown_seconds"] = cooldown(
                        row["seconds"], float(config.get("cooldown_factor", 0.0)))
                    history.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    del sequences, mask, reliability, encoded
            except _OOM as error:
                record["status"] = "HALT"
                record["failure"] = f"OOM: {error}"
                (out_dir / "evaluation.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8")
                raise SystemExit(1)

        record["history"] = history
        checkpoint = out_dir / "checkpoint"
        model.save_pretrained(str(checkpoint))
        record["checkpoint"] = str(checkpoint)

    rows = math500_subset(int(config["evaluation"]["n_problems"]),
                          config["evaluation"]["levels"])
    record["evaluation"] = evaluate_math500(
        model, tokenizer, rows, generation,
        float(config.get("cooldown_factor", 0.0)), config)
    record["evaluation"]["unique_ids"] = [r["unique_id"] for r in rows]
    record["status"] = "OK"

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest(config, {}), indent=2), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"{tag}: accuracy {record['evaluation']['accuracy']*100:.2f}% "
          f"boxed {record['evaluation']['boxed_rate']*100:.1f}% "
          f"({record['evaluation']['seconds']:.0f}s eval)")


if __name__ == "__main__":
    main()
