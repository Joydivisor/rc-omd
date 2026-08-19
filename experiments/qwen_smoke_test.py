"""Q3 engineering smoke test for `qwen-v1-smoke`.

Loads the pinned checkpoint, attaches LoRA, generates, runs a short optimizer
loop, and round-trips a checkpoint, recording every quantity the Q3 task list
requires. **No scientific conclusion may be drawn from this script.** A failure
here is a HALT -- an engineering fact -- and never an algorithmic NO-GO.

Resource discipline is deliberate and not incidental. The host is a laptop with
roughly 8 GB of VRAM whose operator has asked that the machine not be pushed to
its limits, so the process caps its own CUDA allocation, keeps batches and
generation lengths small, samples GPU temperature, and aborts cleanly on OOM or
a non-finite loss rather than retrying into a wall.

Usage:  python experiments/qwen_smoke_test.py --config configs/qwen_smoke.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent

# torch raises torch.OutOfMemoryError; older code paths use torch.cuda.OutOfMemoryError.
# Catching both means the resource guard cannot be bypassed by whichever is raised.
_OOM = tuple({torch.OutOfMemoryError, torch.cuda.OutOfMemoryError})


def gpu_probe() -> dict[str, Any]:
    """Temperature and memory from nvidia-smi; never fatal if unavailable."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        temperature, used, utilisation = (int(x.strip()) for x in out.split(","))
        return {"temperature_c": temperature, "memory_used_mib": used,
                "utilisation_pct": utilisation}
    except Exception as error:  # noqa: BLE001 - diagnostics must never crash the run
        return {"error": f"{type(error).__name__}: {error}"}


def peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    record: dict[str, Any] = OrderedDict()
    record["protocol_id"] = config["protocol_id"]
    record["phase"] = config["phase"]
    record["caveat"] = (
        "Engineering feasibility only. Failure is HALT, not algorithmic NO-GO."
    )
    record["environment"] = OrderedDict([
        ("python", platform.python_version()),
        ("platform", platform.platform()),
        ("torch", torch.__version__),
        ("cuda_runtime", torch.version.cuda),
        ("transformers", transformers.__version__),
        ("gpu", torch.cuda.get_device_name(0)),
        ("gpu_capability", list(torch.cuda.get_device_capability(0))),
        ("gpu_total_mib", torch.cuda.get_device_properties(0).total_memory / 1024**2),
    ])
    record["gpu_before"] = gpu_probe()

    guard = config["resource_guard"]
    fraction = float(guard["cuda_memory_fraction"])
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    record["resource_guard"] = OrderedDict([
        ("cuda_memory_fraction", fraction),
        ("effective_cap_mib",
         fraction * torch.cuda.get_device_properties(0).total_memory / 1024**2),
    ])

    torch.manual_seed(int(config["training"]["seed"]))
    failures: list[str] = []

    model_spec = config["model"]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=model_spec["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        dtype=getattr(torch, model_spec["dtype"]),
        attn_implementation=model_spec.get("attn_implementation", "sdpa"),
    ).to("cuda")
    load_seconds = time.perf_counter() - started
    record["load"] = OrderedDict([
        ("seconds", load_seconds),
        ("model_id", model_spec["id"]),
        ("revision", model_spec["revision"]),
        ("resolved_dtype", str(next(model.parameters()).dtype)),
        ("peak_mib_after_load", peak_mib()),
    ])

    lora = LoraConfig(**config["lora"])
    model = get_peft_model(model, lora)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record["parameters"] = OrderedDict([
        ("total", total), ("trainable", trainable),
        ("trainable_fraction", trainable / total),
    ])

    # --- generation -------------------------------------------------------
    generation = config["generation"]
    prompts = list(config["prompts"])
    messages = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        messages, return_tensors="pt", padding=True, truncation=True,
        max_length=int(generation["max_prompt_tokens"]),
    ).to("cuda")

    model.eval()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=bool(generation["do_sample"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                num_return_sequences=int(generation["completions_per_prompt"]),
                pad_token_id=tokenizer.pad_token_id,
            )
    except _OOM as error:
        failures.append(f"OOM during generation: {error}")
        record["failures"] = failures
        record["verdict"] = "HALT_OOM_GENERATION"
        _write(config, record)
        sys.exit(1)
    generate_seconds = time.perf_counter() - started

    new_tokens = int((generated.shape[1] - batch["input_ids"].shape[1])
                     * generated.shape[0])
    sample = tokenizer.decode(
        generated[0][batch["input_ids"].shape[1]:], skip_special_tokens=True
    )
    record["generation"] = OrderedDict([
        ("sequences", int(generated.shape[0])),
        ("new_tokens_total", new_tokens),
        ("seconds", generate_seconds),
        ("tokens_per_second", new_tokens / generate_seconds),
        ("peak_mib_after_generate", peak_mib()),
        ("sample_completion_head", sample[:400]),
    ])
    record["gpu_after_generate"] = gpu_probe()

    # --- training loop ----------------------------------------------------
    training = config["training"]
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training["learning_rate"]),
    )

    sequences = generated[: int(training["microbatch_sequences"])]
    labels = sequences.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    losses: list[float] = []
    step_times: list[float] = []
    for step in range(int(training["optimizer_steps"])):
        started = time.perf_counter()
        try:
            out = model(input_ids=sequences, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                failures.append(f"non-finite loss at step {step}: {loss.item()}")
                if guard["abort_on_nonfinite"]:
                    break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except _OOM as error:
            failures.append(f"OOM at optimizer step {step}: {error}")
            break
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        losses.append(float(loss.item()))

    record["training"] = OrderedDict([
        ("steps_requested", int(training["optimizer_steps"])),
        ("steps_completed", len(losses)),
        ("loss_first", losses[0] if losses else None),
        ("loss_last", losses[-1] if losses else None),
        ("losses", losses),
        ("seconds_per_step_mean", sum(step_times) / len(step_times) if step_times else None),
        ("seconds_per_step_max", max(step_times) if step_times else None),
        ("peak_mib_after_training", peak_mib()),
        ("any_nonfinite_loss", any(not torch.isfinite(torch.tensor(x)) for x in losses)),
    ])
    record["gpu_after_training"] = gpu_probe()

    # --- checkpoint round trip -------------------------------------------
    checkpoint = Path(config["output_directory"]) / "adapter_checkpoint"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))

    model.eval()
    with torch.no_grad():
        before = float(model(input_ids=sequences, labels=labels).loss.item())

    # The trained model MUST be released before the restore copy is built.
    # Holding both means two full sets of base weights resident at once, which
    # is ~6.2 GB of the 8 GB card before a single activation is allocated.
    sequences = sequences.clone()
    labels = labels.clone()
    del model, optimizer, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    record["checkpoint_free"] = OrderedDict([
        ("mib_allocated_after_release", torch.cuda.memory_allocated() / 1024**2),
        ("gpu", gpu_probe()),
    ])

    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        model_spec["id"], revision=model_spec["revision"],
        dtype=getattr(torch, model_spec["dtype"]),
        attn_implementation=model_spec.get("attn_implementation", "sdpa"),
    ).to("cuda")
    restored = PeftModel.from_pretrained(base, str(checkpoint)).eval()
    with torch.no_grad():
        after = float(restored(input_ids=sequences, labels=labels).loss.item())

    delta = abs(before - after)
    if delta > 1e-4:
        failures.append(f"checkpoint restore loss mismatch: {delta:.3e}")
    record["checkpoint"] = OrderedDict([
        ("path", str(checkpoint)),
        ("loss_before_save", before),
        ("loss_after_restore", after),
        ("absolute_difference", delta),
        ("consistent", delta <= 1e-4),
    ])

    record["peak_mib_overall"] = peak_mib()
    record["gpu_final"] = gpu_probe()
    record["failures"] = failures
    record["verdict"] = "PASS" if not failures else "HALT_ENGINEERING_FAILURE"
    _write(config, record)
    print(f"verdict: {record['verdict']}")
    print(f"peak VRAM {record['peak_mib_overall']:.0f} MiB of "
          f"{record['environment']['gpu_total_mib']:.0f} MiB")


def _write(config: dict[str, Any], record: dict[str, Any]) -> None:
    out = Path(config["output_directory"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
