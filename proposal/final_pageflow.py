from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\setlength{\parskip}{5pt}"
new = r"\setlength{\parskip}{3pt}"
if old not in text:
    raise RuntimeError("Paragraph spacing setting not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
