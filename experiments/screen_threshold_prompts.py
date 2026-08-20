"""Screen MATH training prompts for THRESHOLD difficulty.

GRPO's group-relative advantage is exactly zero when every completion in a group
scores alike, so such a group contributes no gradient at all. The GSM8K run had
88% such groups and the level-4/5 MATH probe still had 81%: hard problems mostly
add all-wrong groups, which are as useless as all-right ones.

This screens candidates with the base model under training sampling conditions
and keeps only prompts where ``1 <= correct <= G-1``. Those are the prompts at
the model's decision boundary, and they are the only ones that carry signal.

The pool is shared by every branch and seed, so screening cannot advantage
either arm: both train on exactly the same prompts in the same order.
"""

from __future__ import annotations

import argparse, json, random, time
from collections import OrderedDict
from pathlib import Path

import torch

from experiments.math500_eval import extract_boxed, score_completion
from experiments.qwen_q3 import load

PROMPT = ("Solve the mathematics problem.\n"
          "Show concise reasoning and put your final answer in \boxed{}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--candidates", type=int, default=320)
    ap.add_argument("--target", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(
        float(cfg["resource_guard"]["cuda_memory_fraction"]), 0)

    from datasets import get_dataset_config_names, load_dataset
    rows = []
    for name in get_dataset_config_names("EleutherAI/hendrycks_math"):
        d = load_dataset("EleutherAI/hendrycks_math", name, split="train")
        for p, l, s in zip(d["problem"], d["level"], d["solution"]):
            if l in ("Level 3", "Level 4", "Level 5"):
                gold = extract_boxed(s)
                if gold:
                    rows.append({"problem": p, "level": l, "gold": gold})
    random.Random(int(cfg["screening"]["pool_seed"])).shuffle(rows)
    rows = rows[: a.candidates]

    tok, model = load(cfg)
    model.eval()
    G = int(cfg["generation"]["completions_per_prompt"])
    maxnew = int(cfg["generation"]["max_new_tokens"])
    batch = int(cfg["screening"]["prompt_batch"])

    kept, screened, started = [], 0, time.perf_counter()
    for i in range(0, len(rows), batch):
        if len(kept) >= a.target:
            break
        chunk = rows[i:i + batch]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{PROMPT}\n\n{r['problem']}"}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=int(cfg["generation"]["max_prompt_tokens"])).to("cuda")
        torch.manual_seed(int(cfg["screening"]["pool_seed"]) + i)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=maxnew, do_sample=True,
                                 temperature=float(cfg["generation"]["temperature"]),
                                 top_p=1.0, num_return_sequences=G,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for j, r in enumerate(chunk):
            n_correct = int(sum(score_completion(comps[j * G + k], r["gold"])
                                for k in range(G)))
            screened += 1
            if 1 <= n_correct <= G - 1:
                kept.append({**r, "base_correct_of_G": n_correct})
        del out, enc
        print(f"  screened {screened}/{len(rows)} kept {len(kept)} "
              f"({time.perf_counter()-started:.0f}s)", flush=True)

    record = OrderedDict([
        ("source", "EleutherAI/hendrycks_math train, Levels 3-5"),
        ("selection", f"base model at threshold: 1 <= correct <= {G-1} of G={G}"),
        ("pool_seed", cfg["screening"]["pool_seed"]),
        ("candidates_screened", screened),
        ("kept", len(kept)),
        ("yield", len(kept) / max(screened, 1)),
        ("seconds", time.perf_counter() - started),
        ("prompts", kept),
    ])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"kept {len(kept)}/{screened} = {record['yield']*100:.0f}% yield "
          f"in {record['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
