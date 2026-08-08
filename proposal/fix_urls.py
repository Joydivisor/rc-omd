from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\usepackage{enumitem}" + "\n" + r"\usepackage{hyperref}"
new = r"\usepackage{enumitem}" + "\n" + r"\usepackage{xurl}" + "\n" + r"\usepackage{hyperref}"
if old not in text:
    raise RuntimeError("Package insertion point not found")
path.write_text(text.replace(old, new), encoding="utf-8")
