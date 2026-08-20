"""Dataset splitting, prompting and answer parsing for the unified team protocol.

Implements Sections 3 and 4 of `docs/QWEN_TEAM_PROTOCOL.md`: a deterministic
SHA-256 bucket split of `openai/gsm8k` `main`, the shared prompt template, and
the protocol's answer-parsing rules.

The split is a pure function of the question text, so it is identical on every
machine and every run and cannot drift with execution order, shuffling, or
model output. That is the point of hashing rather than sampling.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

PROMPT_TEMPLATE = (
    "Solve the grade-school mathematics problem.\n"
    "Show concise reasoning and end with a line of the form #### <number>."
)

DEV_BUCKET_MAX = 999          # buckets 0-999      -> development
TRAIN_BUCKET_MIN = 1000       # buckets 1000-9999  -> training
N_BUCKETS = 10000

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def question_bucket(question: str) -> int:
    """Stable bucket in [0, 9999) from the SHA-256 of the question text."""

    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % N_BUCKETS


def split_of(question: str) -> str:
    return "development" if question_bucket(question) <= DEV_BUCKET_MAX else "training"


def partition(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split GSM8K `train` rows into development and training by hash bucket."""

    out: dict[str, list[dict[str, Any]]] = {"development": [], "training": []}
    for row in rows:
        out[split_of(row["question"])].append(row)
    return out


def build_prompt(question: str) -> list[dict[str, str]]:
    """Chat messages for the Instruct checkpoint, identical for both branches."""

    return [{"role": "user", "content": f"{PROMPT_TEMPLATE}\n\n{question.strip()}"}]


def parse_number(text: str) -> str | None:
    """Protocol Section 4: prefer the number after `####`, else the last number.

    Commas are stripped and a trailing decimal point removed, so `3,450.` and
    `3450` compare equal. Returns None when nothing parses, which the caller
    must score as incorrect rather than discard.
    """

    if text is None:
        return None
    cleaned = text.replace("$", "")
    if "####" in cleaned:
        tail = cleaned.split("####")[-1]
        matches = _NUMBER.findall(tail)
        if matches:
            return _normalize(matches[0])
    matches = _NUMBER.findall(cleaned)
    return _normalize(matches[-1]) if matches else None


def _normalize(raw: str) -> str | None:
    value = raw.replace(",", "").rstrip(".")
    if value in ("", "-"):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    # integers and integral decimals must compare equal: 10 == 10.0
    return str(int(number)) if number == int(number) else repr(number)


def answers_match(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False


def protocol_reward(completion: str, gold_answer_field: str) -> float:
    """1.0 for an exact numeric match of the final answer, else 0.0."""

    return 1.0 if answers_match(parse_number(completion),
                                parse_number(gold_answer_field)) else 0.0


def parse_succeeded(completion: str) -> bool:
    """Used by the Q3b answer-parsing success-rate gate."""

    return parse_number(completion) is not None
