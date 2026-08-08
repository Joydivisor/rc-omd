from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\textbf{To be updated after the final word-count audit.}"
new = r"\textbf{718 words}"
if old not in text:
    raise RuntimeError("Word-count placeholder not found")
path.write_text(text.replace(old, new), encoding="utf-8")
