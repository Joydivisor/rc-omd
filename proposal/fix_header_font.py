from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")
text = text.replace(r"\fancyhead[L]{\small Team PRISM --- Project Proposal}",
                    r"\fancyhead[L]{Team PRISM --- Project Proposal}")
text = text.replace(r"\fancyhead[R]{\small RL-GROUP 1}",
                    r"\fancyhead[R]{RL-GROUP 1}")
text = text.replace(r"\fancyfoot[C]{\small\thepage}",
                    r"\fancyfoot[C]{\thepage}")
path.write_text(text, encoding="utf-8")
