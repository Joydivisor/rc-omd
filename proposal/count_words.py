import re
from pathlib import Path

tex = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")

def section(name: str, next_name: str) -> str:
    start = tex.index(r"\section*{" + name + "}")
    end = tex.index(r"\section*{" + next_name + "}", start)
    return tex[start:end]

def visible_words(source: str) -> list[str]:
    source = re.sub(r"%.*", " ", source)
    source = re.sub(r"\$.*?\$", " ", source, flags=re.S)
    source = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", source)
    source = re.sub(r"\\(?:ganttcell|blankcell|rule|cellcolor)\b(?:\[[^]]*\])?(?:\{[^{}]*\})*", " ", source)
    source = re.sub(r"\\section\*\{[^}]*\}", " ", source)
    source = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^]]*\])?", " ", source)
    source = source.replace("~", " ").replace("--", "-")
    source = source.replace("&", " ").replace("\\", " ")
    source = source.replace("{", " ").replace("}", " ")
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", source)

parts = {
    "Project Summary": section("Project Summary", "Project Significance and Contribution to the Field"),
    "Significance": section("Project Significance and Contribution to the Field", "Project Timeline and Task Allocations"),
    "Timeline and allocations": section("Project Timeline and Task Allocations", "Bibliography"),
}

total = 0
for label, source in parts.items():
    count = len(visible_words(source))
    total += count
    print(f"{label}: {count}")
print(f"Total: {total}")
