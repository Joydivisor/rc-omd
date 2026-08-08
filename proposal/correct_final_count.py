from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\textbf{706 words}"
new = r"\textbf{710 words}"
if old not in text:
    raise RuntimeError("Expected word count not found")
path.write_text(text.replace(old, new), encoding="utf-8")
