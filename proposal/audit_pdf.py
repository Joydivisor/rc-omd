import re
from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
pdf_path = root / "outputs" / "Team_PRISM_Project_Proposal.pdf"
tex_path = root / "outputs" / "Team_PRISM_Project_Proposal.tex"
reader = PdfReader(str(pdf_path))
tex = tex_path.read_text(encoding="utf-8")
texts = [(page.extract_text() or "") for page in reader.pages]
all_text = "\n".join(texts)

print("PDF_PAGES", len(reader.pages))
for i, page in enumerate(reader.pages, 1):
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    print(f"PAGE_{i}_MM {width * 25.4 / 72:.2f} x {height * 25.4 / 72:.2f} TEXT_CHARS {len(texts[i-1].strip())}")

title = "Adaptive Credit-Weighted Online Mirror Descent for Sparse Verifiable Reward Reinforcement Learning"
title_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", title)
print("TITLE_WORDS", len(title_words), title_words)

required = [
    "Project Title",
    "Project Summary",
    "Project Significance and Contribution to the Field",
    "Project Timeline and Task Allocations",
    "Bibliography",
    "Total Word Count (excluding title and bibliography)",
    "Evaluation Sheet",
    "Final Mark",
    "Further Comments",
    "Team PRISM",
    "RL-GROUP 1",
    "HUANG YUXUAN",
    "LIU YIPU",
    "XIE HAOXIANG",
    "ZHONG YIFAN",
    "LI YUFEI",
    "750 words",
]
missing = [item for item in required if item not in all_text]
print("MISSING_REQUIRED_TEXT", missing)

font_records = {}
for page in reader.pages:
    fonts = page.get("/Resources", {}).get("/Font", {})
    for key, ref in fonts.items():
        font = ref.get_object()
        base = str(font.get("/BaseFont", ""))
        subtype = str(font.get("/Subtype", ""))
        embedded = False
        candidates = [font]
        descendants = font.get("/DescendantFonts", [])
        candidates += [item.get_object() for item in descendants]
        for candidate in candidates:
            descriptor = candidate.get("/FontDescriptor")
            if descriptor:
                descriptor = descriptor.get_object()
                if any(name in descriptor for name in ("/FontFile", "/FontFile2", "/FontFile3")):
                    embedded = True
        font_records[(base, subtype)] = font_records.get((base, subtype), False) or embedded
for (base, subtype), embedded in sorted(font_records.items()):
    print("FONT", base, subtype, "EMBEDDED", embedded)

checks = {
    "a4paper": r"\documentclass[11pt,a4paper]" in tex,
    "margins_2_54cm": r"margin=2.54cm" in tex,
    "times_new_roman": r"\setmainfont{Times New Roman}" in tex,
    "line_spacing_1_15": r"\setstretch{1.15}" in tex,
    "no_small_body_commands": r"\small" not in tex and r"\scriptsize" not in tex and r"\footnotesize" not in tex,
    "instructor_notice": "This section is to be completed by the Instructor(s). Please do not delete." in tex,
}
for key, passed in checks.items():
    print("SOURCE_CHECK", key, passed)

placeholder_patterns = [r"\[Name\]", r"PLACEHOLDER", r"TODO", r"To be updated", r"e\.g\."]
print("PLACEHOLDER_HITS", [pattern for pattern in placeholder_patterns if re.search(pattern, tex, re.I)])
