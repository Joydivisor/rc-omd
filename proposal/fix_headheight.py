from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\pagestyle{fancy}"
new = r"\setlength{\headheight}{14pt}" + "\n" + r"\pagestyle{fancy}"
if old not in text:
    raise RuntimeError("Header insertion point not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
