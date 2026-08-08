from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
replacements = {
    "Scope, literature and success criteria": "Scope and literature",
    "Environment, oracle credit and baselines": "Environment and baselines",
    "RC-OMD theory and implementation": "RC-OMD development",
    "Main experiments and ablations": "Experiments and ablations",
    "Analysis, report and presentation": "Analysis and reporting",
    r"\textbf{718 words}": r"\textbf{706 words}",
}
for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"Expected text not found: {old}")
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
