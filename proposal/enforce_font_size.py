from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
text = text.replace("\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}",
                    "\\normalsize\n\\setlength{\\tabcolsep}{1.55pt}")
text = text.replace("\\section*{Bibliography}\n\n\\small\n", "\\section*{Bibliography}\n\n")
text = text.replace("\n\\normalsize\n\\section*{Total Word Count", "\n\\section*{Total Word Count")
path.write_text(text, encoding="utf-8")
