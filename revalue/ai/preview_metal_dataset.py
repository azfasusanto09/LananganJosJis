from pathlib import Path
from PIL import Image, ImageDraw
import math

folder = Path("dataset/revalue_6class_v2/Metal")
files = sorted(folder.glob("*"))

images = []
for path in files:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((140, 110))
        images.append((path.name, img.copy()))
    except Exception:
        pass

cols = 6
cell_w, cell_h = 160, 145
rows = math.ceil(len(images) / cols)

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white",
)
draw = ImageDraw.Draw(sheet)

for i, (name, img) in enumerate(images):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(img, (x + 10, y + 5))
    draw.text((x + 5, y + 118), name, fill="black")

output = Path("metal_dataset_preview.jpg")
sheet.save(output, quality=85)

print(f"Jumlah gambar: {len(images)}")
print(f"Kolase tersimpan di: {output.resolve()}")