from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
old = r"\begin{tabularx}{\textwidth}{>{\bfseries}p{3.0cm}X}"
new = r"\begin{tabularx}{\textwidth}{>{\bfseries\raggedright\arraybackslash}p{3.0cm}X}"
if old not in text:
    raise RuntimeError("Metadata table signature not found")
path.write_text(text.replace(old, new), encoding="utf-8")
