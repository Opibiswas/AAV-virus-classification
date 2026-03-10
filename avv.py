# =========================
# 0. Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 1. Imports
# =========================
import os, copy, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

# =========================
# 2. Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 3. Paths
# =========================
data_dir = "/content/drive/MyDrive/train_val_photos"
output_dir = "/content/drive/MyDrive/AAV_model_outputs_improved_100mV_1s_trans"
os.makedirs(output_dir, exist_ok=True)

print("Dataset path found:", data_dir)
print("Class folders:", os.listdir(data_dir))
print("Output folder:", output_dir)

# =========================
# 4. Dataset with paths
# =========================
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

base_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB"))
])

full_dataset = ImageFolderWithPaths(root=data_dir, transform=base_transform)
class_names = full_dataset.classes
targets = [s[1] for s in full_dataset.samples]
indices = list(range(len(full_dataset)))

print("Classes:", class_names)
print("Total images:", len(indices))

train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=targets
)

print("Train images:", len(train_idx))
print("Val images:", len(val_idx))

# =========================
# 5. Class weights
# =========================
label_counts = Counter(targets)
num_classes = len(class_names)
total_samples = len(targets)

class_weights = []
for i in range(num_classes):
    class_weights.append(total_samples / (num_classes * label_counts[i]))
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:", class_weights)

# =========================
# 6. Transform subset
# =========================
class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, path = self.dataset[self.indices[i]]
        if self.transform:
            x = self.transform(x)
        return x, y, path

# =========================
# 7. Transforms
# =========================
def get_transforms(model_name):
    size = 299 if model_name == "InceptionV3" else 224

    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# =========================
# 8. Build improved models
# =========================
def build_model(model_name, num_classes):
    if model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():   # last block unfreeze
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "VGG19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[28:].parameters():  # last conv block
            p.requires_grad = True
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif model_name == "InceptionV3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.Mixed_7a.parameters():
            p.requires_grad = True
        for p in model.Mixed_7b.parameters():
            p.requires_grad = True
        for p in model.Mixed_7c.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)

# =========================
# 9. Inception forward helper
# =========================
def forward_model(model, x, model_name, train_mode=True):
    if model_name != "InceptionV3":
        return model(x)

    out = model(x)
    if train_mode:
        if hasattr(out, "logits"):
            return out.logits, out.aux_logits
        elif isinstance(out, tuple):
            return out
        else:
            return out, None
    else:
        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, tuple):
            return out[0]
        else:
            return out

# =========================
# 10. Train function
# =========================
def train_and_evaluate(model_name, epochs=8, batch_size=16, lr=1e-4):
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    train_tf, val_tf = get_transforms(model_name)
    train_ds = TransformSubset(full_dataset, train_idx, transform=train_tf)
    val_ds = TransformSubset(full_dataset, val_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(model_name, len(class_names))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        tr_loss, tr_preds, tr_labels = 0.0, [], []

        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            if model_name == "InceptionV3":
                out, aux = forward_model(model, x, model_name, train_mode=True)
                loss = criterion(out, y) + (0.4 * criterion(aux, y) if aux is not None else 0)
                preds = out.argmax(1)
            else:
                out = forward_model(model, x, model_name, train_mode=True)
                loss = criterion(out, y)
                preds = out.argmax(1)

            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_preds.extend(preds.detach().cpu().numpy())
            tr_labels.extend(y.detach().cpu().numpy())

        tr_loss /= len(train_ds)
        tr_acc = accuracy_score(tr_labels, tr_preds)

        model.eval()
        va_loss, va_preds, va_labels = 0.0, [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = forward_model(model, x, model_name, train_mode=False)
                loss = criterion(out, y)
                preds = out.argmax(1)

                va_loss += loss.item() * x.size(0)
                va_preds.extend(preds.cpu().numpy())
                va_labels.extend(y.cpu().numpy())

        va_loss /= len(val_ds)
        va_acc = accuracy_score(va_labels, va_preds)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())

    total_time = time.time() - start
    model.load_state_dict(best_wts)

    # final eval
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device, non_blocking=True)
            out = forward_model(model, x, model_name, train_mode=False)
            preds = out.argmax(1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(y.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="macro", zero_division=0
    )
    acc = accuracy_score(final_labels, final_preds)
    cm = confusion_matrix(final_labels, final_preds)

    model_path = os.path.join(output_dir, f"{model_name}_improved_best.pth")
    torch.save(model.state_dict(), model_path)

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "training_time_sec": total_time,
        "history": history,
        "confusion_matrix": cm,
        "class_names": class_names,
        "model_path": model_path
    }

# =========================
# 11. Run experiments
# =========================
results = []
for model_name in ["ResNet50", "VGG19", "InceptionV3"]:
    results.append(train_and_evaluate(model_name, epochs=8, batch_size=16, lr=1e-4))

# =========================
# 12. Comparison table
# =========================
comparison_df = pd.DataFrame([{
    "Model": r["model_name"],
    "Best Val Acc": round(r["best_val_acc"], 4),
    "Final Val Acc": round(r["final_val_acc"], 4),
    "Precision (Macro)": round(r["precision_macro"], 4),
    "Recall (Macro)": round(r["recall_macro"], 4),
    "F1-score (Macro)": round(r["f1_macro"], 4),
    "Training Time (sec)": round(r["training_time_sec"], 2)
} for r in results])

print("\nModel Comparison")
print(comparison_df)

csv_path = os.path.join(output_dir, "improved_model_comparison.csv")
comparison_df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

# =========================
# 13. Curves
# =========================
for r in results:
    hist = r["history"]
    ep = range(1, len(hist["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, hist["train_loss"], marker='o', label="Train")
    plt.plot(ep, hist["val_loss"], marker='o', label="Val")
    plt.title(f"{r['model_name']} Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ep, hist["train_acc"], marker='o', label="Train")
    plt.plot(ep, hist["val_acc"], marker='o', label="Val")
    plt.title(f"{r['model_name']} Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

# =========================
# 14. Confusion matrices
# =========================
for r in results:
    cm = r["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{r['model_name']} Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(r["class_names"]))
    plt.xticks(ticks, r["class_names"], rotation=45)
    plt.yticks(ticks, r["class_names"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()
