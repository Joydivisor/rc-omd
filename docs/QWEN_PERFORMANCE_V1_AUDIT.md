# Qwen Performance V1: model and data audit

Protocol ID: `qwen-performance-v1` (Q2)

Status: **frozen as of this commit**, except the four items explicitly marked
**pending Q3 measurement**. Definitions are in
[`QWEN_PERFORMANCE_V1_DEFINITIONS.md`](QWEN_PERFORMANCE_V1_DEFINITIONS.md).

## Model

| item | value |
|---|---|
| Hugging Face ID | `Qwen/Qwen2.5-Math-1.5B` |
| revision (pinned) | `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` |
| tokenizer | the tokenizer shipped at that revision; no substitution |
| vocabulary | 151936 |
| layers / hidden / kv-heads / head-dim | 28 / 1536 / 2 / 128 |
| `max_position_embeddings` | 4096 |
| dtype | bfloat16 |

Both branches load **this exact revision**, verified by hash at run start. A
mismatch is a **HALT**.

### There is no `Qwen2.5-Math-0.5B`

Checked against the hub: the Qwen2.5-Math family is **1.5B, 7B and 72B only**
(plus Instruct/PRM/RM variants). The 0.5B math checkpoint named as a
possibility in the plan does not exist, so that fallback is unavailable. The
only smaller alternative is `Qwen/Qwen2.5-0.5B-Instruct`, which is a general
model with no mathematical specialisation.

**Selected: `Qwen2.5-Math-1.5B` with LoRA.** It is the original target, it is
math-specialised so clean accuracy sits far enough above floor for a 2-point
non-inferiority margin to be meaningful, and the memory budget below shows it
fits. The 0.5B general model would need `bitsandbytes` 8-bit optimisers, which
are a known risk on Windows with a Blackwell GPU, and would trade a real
engineering risk for a weaker scientific setting.

### Contamination disclosure

**Whether this checkpoint was trained on GSM8K's training split is not
established.** Qwen2.5-Math was trained on a large mathematical corpus that
plausibly includes GSM8K-style data and may include GSM8K itself. This must be
stated in any report.

The consequence is bounded and must be stated with equal precision:
contamination inflates **absolute** accuracy and makes it incomparable with
published GSM8K numbers, but it does **not** invalidate the comparison this
protocol makes. Both branches start from the identical checkpoint and are
compared to each other, so `Delta_acc` and `RobustnessDrop` remain
interpretable. Absolute accuracy may not be quoted as a GSM8K result.

## LoRA

| item | value |
|---|---|
| enabled | yes, required -- see budget |
| rank `r` | 16 |
| `alpha` | 32 |
| dropout | 0.0 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| base weights | frozen, bf16 |

Full fine-tuning is **not feasible**: 1.5B parameters need ~3.1 GB bf16 weights
plus ~3.1 GB gradients plus ~12.3 GB fp32 AdamW state, against 8.0 GB of VRAM.

LoRA restricts the parameter geometry, which was a scientific objection when
this was a mechanism experiment. **Under the Q0 reframing it is not one**: this
protocol asks whether RWP-OMD improves the accuracy/robustness trade-off for
this checkpoint under this training setup, and LoRA is part of that setup rather
than a confound to it. The claim is correspondingly narrow and the report must
keep it narrow.

## Memory budget (measured configs, arithmetic; Q3 must verify empirically)

Available: **8151 MiB**.

| component | GB |
|---|---:|
| bf16 base weights | 3.08 |
| LoRA adapters + optimiser state | ~0.10 |
| KV cache, 8 rollouts x 640 tokens | 0.15 |
| logits + `pi_old` logits, 1024-token microbatch | 0.62 |
| logit gradients | ~0.31 |
| activations with gradient checkpointing | ~0.5 |
| CUDA context and fragmentation | ~0.5 |
| **total** | **~5.3** |

Headroom ~2.8 GB. The full-vocabulary cross-entropy of D4 costs
`tokens x 151936 x 2 bytes` and is **independent of model size**, so it is
controlled by the microbatch token count rather than by choosing a smaller
model. Microbatching, not truncation, is therefore the primary mitigation.

**Pending Q3 measurement**: peak VRAM under real generation and backward; whether
the full-vocabulary cross-entropy fits at the frozen lengths; throughput; and
whether the top-`k` fallback of D4 is needed at all.

## Lengths

| item | value |
|---|---:|
| max prompt tokens | 512 |
| max generated tokens | 640 |
| total context used | 1152 of 4096 |

Truncated generations are scored as incorrect, not discarded, so length
truncation cannot silently favour either branch.

## Data

Two tiers, both frozen before any evaluation.

**Tier 1 -- clean benchmark.** `openai/gsm8k`, config `main`. The `train` split
supplies the RL prompts and the development split; the `test` split is the
Tier 1 evaluation set and is not inspected before formal evaluation.

**Tier 2 -- distractor diagnostic.** `voidful/GSM-IC`, the GSM-IC set of Shi et
al., which appends irrelevant sentences to GSM8K problems and therefore supplies
distractor-perturbed items with known clean counterparts. Verified present on
the hub. Its licence and exact schema are **pending Q3 verification**; if the
clean/perturbed pairing cannot be recovered per item, `RobustnessDrop` is not
computable as defined and this is a **HALT**, not a NO-GO.

**Development split.** 500 problems drawn from `gsm8k/train` by
`numpy.random.default_rng(20260825)`, disjoint from the RL prompt pool. All Q4
hyperparameter selection uses this split and nothing else.

**Test discipline.** Neither Tier 1 test nor Tier 2 is inspected, plotted, or
used for any selection before the Q5 freeze. Formal evaluation runs **once**.

## Prompt template and answer extraction

Frozen, identical for both branches:

```
Question: {question}
Answer: Let's think step by step.
```

Extraction takes the **last** integer or decimal appearing in the completion,
after stripping commas and currency symbols, and compares it to the GSM8K gold
answer parsed from after `####`. A completion with no parseable number is
scored incorrect.

This is a deliberately plain rule. It is applied identically to both branches
and to both tiers, so extraction quality cancels in every paired comparison
reported.

## Software stack

`torch 2.11.0+cu128` is present. `transformers`, `datasets`, `accelerate` and
`peft` are **not installed** and are Q3's first task. `trl` is not required --
the GRPO loop is implemented directly, because D5's exact-equivalence test needs
both branches to share one code path with reliability as the only difference.

## What Q2 does not settle

- `eta`, `lambda`, `warmup`, `c`, `r_floor`, `G`, temperature, learning rates --
  all Q4, development split only.
- Every item marked **pending Q3 measurement** above.
