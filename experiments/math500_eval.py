"""MATH-500 answer extraction and scoring.

MATH-500 answers are LaTeX, not plain numbers -- `\\left( 3, \\frac{\\pi}{2}
\\right)`, `3\\sqrt{13}`, `\\text{Evelyn}` -- and only about 64% are numeric. The
GSM8K "last number in the text" rule would therefore score roughly a third of
the benchmark wrong regardless of what the model produced, so scoring here is
`\\boxed{}` extraction plus LaTeX normalization, which is what the standard
harnesses do.

The extractor is validated against the dataset's own gold solutions: recovering
`answer` from `solution` is a lower bound on the extractor's competence, and a
poor recovery rate means the harness is broken rather than the model.
"""

from __future__ import annotations

import re
from typing import Any

_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str) -> str | None:
    """Contents of the LAST `\\boxed{...}`, brace-matched rather than regexed.

    Nested braces are common (`\\boxed{\\frac{1}{2}}`), so a naive
    non-greedy regex truncates at the first close brace and silently produces
    wrong answers.
    """

    if not text:
        return None
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return text[start:].strip()   # unbalanced: take the tail


_STRIP_PATTERNS = [
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\,"), ""),
    (re.compile(r"\\;"), ""),
    (re.compile(r"\\ "), " "),
    (re.compile(r"\\\$"), ""),
    (re.compile(r"\$"), ""),
    (re.compile(r"\\%"), ""),
    (re.compile(r"%"), ""),
    (re.compile(r"\\text\s*{([^}]*)}"), r"\1"),
    (re.compile(r"\\mbox\s*{([^}]*)}"), r"\1"),
    (re.compile(r"\\mathrm\s*{([^}]*)}"), r"\1"),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\tfrac"), r"\\frac"),
    (re.compile(r"\^\s*\\circ"), ""),
    (re.compile(r"\\degree"), ""),
    (re.compile(r"\s+"), ""),
]


def normalize(answer: str | None) -> str | None:
    """Canonical form for string comparison. Conservative by design."""

    if answer is None:
        return None
    text = answer.strip()
    # \frac{a}{b} -> a/b, applied repeatedly for nesting
    for _ in range(3):
        text = re.sub(r"\\frac\s*{([^{}]*)}\s*{([^{}]*)}", r"(\1)/(\2)", text)
    for pattern, replacement in _STRIP_PATTERNS:
        text = pattern.sub(replacement, text)
    text = text.rstrip(".").strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    # trailing zeros: 0.50 -> 0.5, 3.0 -> 3
    if re.fullmatch(r"-?\d+\.\d+", text):
        text = text.rstrip("0").rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
        text = text.replace(",", "")
    return text or None


def answers_equivalent(predicted: str | None, gold: str | None) -> bool:
    """Normalized string match, with a numeric fallback for float formatting."""

    a, b = normalize(predicted), normalize(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def score_completion(completion: str, gold_answer: str) -> float:
    return 1.0 if answers_equivalent(extract_boxed(completion), gold_answer) else 0.0


def extractor_self_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recover `answer` from the dataset's own `solution`.

    This bounds the harness's competence from below. A low rate here means the
    scorer is broken and any model number computed with it is meaningless, so
    this runs before any model is loaded.
    """

    recovered = 0
    boxed_present = 0
    failures: list[dict[str, str]] = []
    for row in rows:
        boxed = extract_boxed(row["solution"])
        if boxed is not None:
            boxed_present += 1
        if answers_equivalent(boxed, row["answer"]):
            recovered += 1
        elif len(failures) < 8:
            failures.append({
                "gold": row["answer"], "extracted": str(boxed),
                "normalized_gold": str(normalize(row["answer"])),
                "normalized_extracted": str(normalize(boxed)),
            })
    total = len(rows)
    return {
        "rows": total,
        "boxed_present_rate": boxed_present / total,
        "gold_recovery_rate": recovered / total,
        "example_failures": failures,
    }
