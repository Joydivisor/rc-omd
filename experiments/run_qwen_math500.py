"""Paired Uniform/RWP GRPO training on GSM8K, evaluated on MATH-500.

One process per (branch, seed). Training prompts come from the GSM8K training
buckets of the hash split; the evaluation target is a frozen MATH-500 subset, so
this measures **transfer** from GSM8K reinforcement learning to MATH rather than
in-distribution improvement, and must be reported that way.

Writes the Section 12 artifacts for every run. Resource discipline matches the
rest of the phase: the process caps its own CUDA allocation, microbatches the
backward pass, and aborts cleanly rather than retrying into a wall.
"""

from __future__ import annotations

import argparse
import gc
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
from experiments.qwen_data import build_prompt, partition, protocol_reward
from experiments.qwen_q3 import load, manifest

_OOM = tuple({torch.OutOfMemoryError, torch.cuda.OutOfMemoryError})

MATH_PROMPT = (
    "Solve the mathematics problem.\n"
    "Show concise reasoning and put your final answer in \\boxed{}."
)


def training_prompts(config, seed: int, needed: int):
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split="train")
    rows = [{"question": q, "answer": a}
            for q, a in zip(data["question"], data["answer"])]
    pool = partition(rows)["training"]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))
    return [pool[int(i)] for i in order[:needed]]


def math500_subset(count: int):
    from datasets import load_dataset

    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"problem": p, "answer": a, "unique_id": u}
            for p, a, u in zip(data["problem"], data["answer"], data["unique_id"])]
    # Deterministic and content-addressed: sorting by unique_id fixes the subset
    # before any model is run and makes it identical on every machine.
    rows.sort(key=lambda r: r["unique_id"])
    return rows[:count]


def evaluate_math500(model, tokenizer, rows, generation) -> dict[str, Any]:
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
    for start in range(0, len(rows), batch_size):
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
    model.config.use_cache = was_cache
    return {
        "n": len(rows),
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
    record["training_data"] = "openai/gsm8k main, training hash buckets"
    record["evaluation_data"] = "HuggingFaceH4/MATH-500, frozen subset by unique_id"
    record["note"] = ("Transfer evaluation: reinforcement learning on GSM8K, "
                      "scored on MATH-500. Not an in-distribution result.")

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
        prompts = training_prompts(config, arguments.seed, steps * per_step)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(training["learning_rate"]),
        )
        history = []
        with metrics_path.open("w", encoding="utf-8") as handle:
            try:
                for step in range(steps):
                    chunk = prompts[step * per_step:(step + 1) * per_step]
                    texts = [
                        tokenizer.apply_chat_template(
                            build_prompt(r["question"]), tokenize=False,
                            add_generation_prompt=True)
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
                        [protocol_reward(c, chunk[i // group_size]["answer"])
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

    rows = math500_subset(int(config["evaluation"]["n_problems"]))
    record["evaluation"] = evaluate_math500(model, tokenizer, rows, generation)
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
