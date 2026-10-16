
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = "dataset/revalue_split_v2/test"
MODEL_PATH = "models/revalue_efficientnet_b0_v2_candidate_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("ReValue AI - Candidate Model Evaluation")
print("=" * 65)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = test_dataset.classes
print(f"\nJumlah gambar test: {len(test_dataset)}")
print("Kelas:", classes)

print("\nMemuat model kandidat...")
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Mendukung format checkpoint dari train_v2.py
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Model berhasil dimuat.")

all_labels = []
all_predictions = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

accuracy = correct / total * 100

print("\n" + "=" * 65)
print("HASIL EVALUASI MODEL KANDIDAT")
print("=" * 65)
print(f"Test images : {total}")
print(f"Accuracy    : {accuracy:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0
))

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)

print("CONFUSION MATRIX")
print("Baris = kelas aktual | Kolom = hasil prediksi")
print("Kelas:", classes)
print(cm)
print("=" * 65)
