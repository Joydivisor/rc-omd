from pathlib import Path
from PIL import Image, ImageDraw
import pypdfium2 as pdfium

base = Path(__file__).resolve().parent
pdf_path = base / "main.pdf"
out_dir = base / "qa_pages"
out_dir.mkdir(exist_ok=True)

pdf = pdfium.PdfDocument(pdf_path)
pages = []
for index in range(len(pdf)):
    image = pdf[index].render(scale=1.5).to_pil().convert("RGB")
    path = out_dir / f"page-{index + 1:02d}.png"
    image.save(path)
    pages.append(image)

thumb_w = 500
margin = 24
label_h = 30
thumbs = []
for image in pages:
    height = round(image.height * thumb_w / image.width)
    thumbs.append(image.resize((thumb_w, height)))

cols = 2
rows = (len(thumbs) + cols - 1) // cols
cell_h = max(image.height for image in thumbs) + label_h
sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin,
                          rows * cell_h + (rows + 1) * margin), "white")
draw = ImageDraw.Draw(sheet)
for index, image in enumerate(thumbs):
    x = margin + (index % cols) * (thumb_w + margin)
    y = margin + (index // cols) * (cell_h + margin)
    draw.text((x, y), f"Page {index + 1}", fill="black")
    sheet.paste(image, (x, y + label_h))
sheet.save(base / "qa_contact_sheet.png")
print(f"pages={len(pages)}")
