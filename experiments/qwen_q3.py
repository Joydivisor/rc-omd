"""Q3a engineering smoke test and Q3b learnability test, per the team protocol.

Q3a (Section 6) checks the machinery: load, adapt, generate, one forward and
backward, one optimizer step, save and reload, and compare logits across the
round trip to 1e-5. Q3b (Section 7) checks the task is learnable at all, on 64
development questions with four completions each.

Both write the mandatory artifacts of Section 12. A failure in either is a
**HALT** -- an engineering fact -- and must never be read as an RWP algorithm
failure.

Usage:
    python experiments/qwen_q3.py --config configs/qwen_team.json --stage q3a
    python experiments/qwen_q3.py --config configs/qwen_team.json --stage q3b
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from experiments.qwen_data import (
    build_prompt,
    parse_number,
    parse_succeeded,
    partition,
    protocol_reward,
)

_OOM = tuple({torch.OutOfMemoryError, torch.cuda.OutOfMemoryError})


def manifest(config: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    import datasets
    import peft
    import transformers

    return OrderedDict([
        ("protocol_id", config["protocol_id"]),
        ("model_id", config["model"]["id"]),
        ("model_revision", config["model"]["revision"]),
        ("dataset", config["data"]["dataset"]),
        ("dataset_config", config["data"]["config"]),
        ("dataset_revision", extra.get("dataset_revision")),
        ("precision", config["model"]["dtype"]),
        ("lora", config["lora"]),
        ("torch", torch.__version__),
        ("transformers", transformers.__version__),
        ("peft", peft.__version__),
        ("datasets", datasets.__version__),
        ("python", platform.python_version()),
        ("gpu", torch.cuda.get_device_name(0)),
        ("gpu_total_mib", torch.cuda.get_device_properties(0).total_memory / 1024**2),
        ("platform", platform.platform()),
    ])


def load(config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"],
        dtype=getattr(torch, spec["dtype"]), attn_implementation="sdpa",
    ).to("cuda")
    model = get_peft_model(model, LoraConfig(**config["lora"]))
    return tokenizer, model


def load_dev_questions(config: dict[str, Any], count: int):
    from datasets import load_dataset

    data = load_dataset(config["data"]["dataset"], config["data"]["config"],
                        split="train")
    rows = [{"question": q, "answer": a}
            for q, a in zip(data["question"], data["answer"])]
    development = partition(rows)["development"]
    return development[:count], len(development), len(rows)


def generate(model, tokenizer, questions, generation, seed, n_completions):
    texts = [tokenizer.apply_chat_template(build_prompt(q["question"]),
                                           tokenize=False, add_generation_prompt=True)
             for q in questions]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=int(generation["max_prompt_tokens"])).to("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **batch, max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=True, temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]), num_return_sequences=n_completions,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    prompt_length = batch["input_ids"].shape[1]
    completions = tokenizer.batch_decode(out[:, prompt_length:],
                                         skip_special_tokens=True)
    truncated = [bool(row[-1] != tokenizer.eos_token_id
                      and row[-1] != tokenizer.pad_token_id) for row in out]
    return out, prompt_length, completions, truncated


def run_q3a(config: dict[str, Any]) -> dict[str, Any]:
    """Section 6. Pass criteria are checked explicitly, not assumed."""

    report: dict[str, Any] = OrderedDict()
    tokenizer, model = load(config)
    questions, _, _ = load_dev_questions(config, 2)
    generation = config["generation"]

    sequences, prompt_length, completions, _ = generate(
        model, tokenizer, questions, generation, config["seeds"]["pilot"], 1
    )
    report["completions_nonempty"] = all(c.strip() for c in completions)
    report["sample_completion"] = completions[0][:300]

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    labels = sequences.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    out = model(input_ids=sequences[:1], labels=labels[:1])
    loss = out.loss
    report["loss"] = float(loss)
    report["loss_finite"] = bool(torch.isfinite(loss))
    loss.backward()

    lora_norm = torch.sqrt(sum(
        (p.grad.float() ** 2).sum() for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    ))
    report["lora_grad_norm"] = float(lora_norm)
    report["lora_grad_positive"] = float(lora_norm) > 0.0
    report["grads_finite"] = all(
        torch.isfinite(p.grad).all() for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        before = model(input_ids=sequences[:1]).logits.float().cpu()

    checkpoint = Path(config["output_directory"]) / "q3a" / "checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))

    # Release before rebuilding: two full base models will not fit in 8 GB.
    import gc

    del model, optimizer, out, loss
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    spec = config["model"]
    base = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], dtype=getattr(torch, spec["dtype"]),
        attn_implementation="sdpa",
    ).to("cuda")
    restored = PeftModel.from_pretrained(base, str(checkpoint)).eval()
    with torch.no_grad():
        after = restored(input_ids=sequences[:1]).logits.float().cpu()

    error = float((before - after).abs().max())
    report["max_logit_error_after_reload"] = error
    report["reload_within_tolerance"] = error <= 1e-5

    checks = ["completions_nonempty", "loss_finite", "grads_finite",
              "lora_grad_positive", "reload_within_tolerance"]
    report["failed_checks"] = [c for c in checks if not report[c]]
    report["verdict"] = "PASS" if not report["failed_checks"] else "HALT"
    report["peak_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    return report


def run_q3b(config: dict[str, Any]) -> dict[str, Any]:
    """Section 7. 64 development questions, four completions each."""

    report: dict[str, Any] = OrderedDict()
    tokenizer, model = load(config)
    n_questions = int(config["q3b"]["questions"])
    n_completions = int(config["q3b"]["completions_per_question"])
    questions, dev_total, all_total = load_dev_questions(config, n_questions)
    report["development_pool"] = dev_total
    report["train_split_rows"] = all_total
    report["questions_used"] = len(questions)

    generation = config["generation"]
    batch_size = int(config["q3b"]["question_batch"])
    all_completions: list[str] = []
    all_truncated: list[bool] = []
    rewards: list[float] = []
    started = time.perf_counter()
    for start in range(0, len(questions), batch_size):
        chunk = questions[start:start + batch_size]
        _, _, completions, truncated = generate(
            model, tokenizer, chunk, generation,
            config["seeds"]["pilot"] + start, n_completions,
        )
        all_completions.extend(completions)
        all_truncated.extend(truncated)
        for index, completion in enumerate(completions):
            rewards.append(protocol_reward(completion,
                                           chunk[index // n_completions]["answer"]))
    elapsed = time.perf_counter() - started

    total = len(all_completions)
    grouped = [rewards[i:i + n_completions]
               for i in range(0, len(rewards), n_completions)]
    varied = sum(1 for g in grouped if len(set(g)) > 1)

    report["total_generations"] = total
    report["seconds"] = elapsed
    report["parse_rate"] = sum(parse_succeeded(c) for c in all_completions) / total
    report["empty_rate"] = sum(1 for c in all_completions if not c.strip()) / total
    report["truncation_rate"] = sum(all_truncated) / total
    report["mean_reward"] = sum(rewards) / total
    report["groups_with_reward_variance"] = varied / len(grouped)
    report["n_groups"] = len(grouped)
    report["peak_mib"] = torch.cuda.max_memory_allocated() / 1024**2

    gates = OrderedDict([
        ("parse_rate>=0.95", report["parse_rate"] >= 0.95),
        ("reward_variance_groups>=0.20", report["groups_with_reward_variance"] >= 0.20),
        ("empty_rate==0", report["empty_rate"] == 0.0),
        ("truncation_rate<=0.05", report["truncation_rate"] <= 0.05),
    ])
    report["gates"] = gates
    report["failed_checks"] = [k for k, ok in gates.items() if not ok]
    report["verdict"] = "PASS" if not report["failed_checks"] else "HALT"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["q3a", "q3b"], required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    torch.cuda.set_per_process_memory_fraction(
        float(config["resource_guard"]["cuda_memory_fraction"]), 0
    )

    try:
        report = run_q3a(config) if arguments.stage == "q3a" else run_q3b(config)
    except _OOM as error:
        report = OrderedDict([("verdict", "HALT"), ("failure", f"OOM: {error}")])

    out = Path(config["output_directory"]) / arguments.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(
        json.dumps(manifest(config, {}), indent=2), encoding="utf-8"
    )
    (out / "evaluation.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
    print(f"{arguments.stage} verdict: {report['verdict']}")
    for key, value in report.items():
        if key in ("sample_completion", "gates"):
            continue
        if isinstance(value, (int, float, bool, str, list)):
            print(f"  {key}: {value}")
    if "gates" in report:
        for key, ok in report["gates"].items():
            print(f"  gate {key}: {ok}")


if __name__ == "__main__":
    main()
