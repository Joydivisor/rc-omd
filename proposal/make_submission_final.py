from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")

summary_anchor = (
    "Because computing resources are not yet confirmed, the core study is designed for modest hardware; "
    "a small-model RLVR demonstration will proceed only if resources and controlled results satisfy a "
    "recorded Go/No-Go decision."
)
summary_replacement = summary_anchor + (
    " To keep conclusions credible, all comparisons will use matched training budgets, multiple random "
    "seeds and predeclared success criteria, with negative results retained."
)

significance_anchor = (
    "Whether or not performance improves, the calibration analysis will identify when local reliability "
    "control is useful and establish a boundary for future RLVR research."
)
significance_replacement = significance_anchor + (
    " The reliability estimate will also be evaluated as a calibrated prediction, making uncertainty "
    "quality an explicit scientific outcome."
)

replacements = {
    summary_anchor: summary_replacement,
    significance_anchor: significance_replacement,
    r"\textbf{710 words}": r"\textbf{750 words}",
}

for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"Expected text not found: {old[:60]}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
