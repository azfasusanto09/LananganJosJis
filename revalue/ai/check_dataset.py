
from pathlib import Path

root = Path("dataset/revalue_split_v2")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for part in ["train", "val", "test"]:
    folder = root / part

    print(f"\n=== {part.upper()} ===")

    if not folder.exists():
        print("Folder tidak ditemukan!")
        continue

    for class_folder in sorted(folder.iterdir()):
        if class_folder.is_dir():
            count = sum(
                1
                for file in class_folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in extensions
            )

            print(f"{class_folder.name}: {count} gambar")
