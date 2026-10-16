
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi
DATA_DIR = Path("dataset/revalue_split_v2")
MODEL_PATH = Path("models/revalue_efficientnet_b0_v2_best.pth")
IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

classes = checkpoint["classes"]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

print("\n=== CONFUSION MATRIX ===")
print("Baris = label sebenarnya, kolom = prediksi")
print("Urutan kelas:", classes)
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
    digits=4,
))

print("\n=== KESALAHAN METAL ===")
metal_index = classes.index("Metal")
mistakes = Counter()

for actual, predicted in zip(all_labels, all_predictions):
    if actual == metal_index and predicted != metal_index:
        mistakes[classes[predicted]] += 1

print("Gambar Metal yang salah diprediksi:", sum(mistakes.values()))
for name, count in mistakes.most_common():
    print(f"  Metal -> {name}: {count}")
