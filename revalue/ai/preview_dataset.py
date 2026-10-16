
from pathlib import Path
import random
from PIL import Image, ImageDraw
import math

ROOT = Path("dataset/revalue_split_v2/train")
CLASSES = ["Plastic", "Paper_Cardboard"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLES_PER_CLASS = 12
THUMB_SIZE = (180, 150)

random.seed(42)

tiles = []

for class_name in CLASSES:
    folder = ROOT / class_name
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    ]

    selected = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

    for path in selected:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)

            tile = Image.new("RGB", (200, 195), "white")
            x = (200 - img.width) // 2
            tile.paste(img, (x, 5))

            draw = ImageDraw.Draw(tile)
            draw.text((5, 160), class_name, fill="black")
            draw.text((5, 177), path.name[:25], fill="black")

            tiles.append(tile)
        except Exception as exc:
            print(f"Gagal membaca {path}: {exc}")

cols = 4
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 200, rows * 195), "#dddddd")

for i, tile in enumerate(tiles):
    x = (i % cols) * 200
    y = (i // cols) * 195
    sheet.paste(tile, (x, y))

output = Path(__file__).resolve().parent / "dataset_preview.jpg"
sheet.save(output, quality=90)

print(f"Preview tersimpan: {output.resolve()}")
print(f"Jumlah gambar preview: {len(tiles)}")
