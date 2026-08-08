from pathlib import Path

path = Path(__file__).with_name("audit_pdf.py")
text = path.read_text(encoding="utf-8")
old = '"710 words",'
new = '"750 words",'
if old not in text:
    raise RuntimeError("Old audit word count not found")
path.write_text(text.replace(old, new), encoding="utf-8")
