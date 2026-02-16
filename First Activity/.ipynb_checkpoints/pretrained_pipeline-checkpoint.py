"""
pretrained_pipeline.py

Reproducible training and evaluation pipeline for digit classification
using a pretrained ResNet18 model.

Pipeline Design
---------------
• Train + Validation split is performed on LT01 (source domain)
• Final evaluation is performed on LT10 (exchanged/target domain)
• Fully reproducible via fixed random seed
• Designed for import into CNN_pretrained.ipynb

Reproducibility Guarantee
-------------------------
All sources of randomness are controlled:
- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- CuDNN deterministic mode
- DataLoader shuffling
- Train/validation split

Author: Your Name
Course: ML Hands-on Activity
"""

import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 0) GLOBAL CONFIG
# ============================================================

SEED = 42
TRAIN_DIR = "ML-Hands-on-Activity/LT01_digits"
TEST_DIR  = "ML-Hands-on-Activity/LT10_digits"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1) REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) DATASET UTILITIES
# ============================================================

def collect_paths(dataset_dir: str):
    """
    Collect all image file paths from a digit dataset directory.

    Expected directory structure:
        dataset_dir/
            0/
            1/
            ...
            9/

    Returns
    -------
    list of (image_path, label)
    """
    dataset_dir = Path(dataset_dir)

    missing = [str(i) for i in range(10) if not (dataset_dir / str(i)).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {dataset_dir}: {missing}"
        )

    items = []
    for label in range(10):
        for p in sorted((dataset_dir / str(label)).glob("*.png")):
            items.append((str(p), label))

    return items


class DigitsDataset(Dataset):
    """
    Custom PyTorch Dataset for digit classification.
    """

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ============================================================
# 3) TRANSFORMS (ImageNet statistics)
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(12),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10),
        shear=(-8, 8),
    ),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


# ============================================================
# 4) DATALOADERS (STRATIFIED SPLIT)
# ============================================================

def make_loaders(items, batch_size: int = 32, seed: int = 42):
    """
    Create stratified train/validation DataLoaders.

    Parameters
    ----------
    items : list
        List of (path, label)
    batch_size : int
    seed : int

    Returns
    -------
    train_dl, val_dl, train_items, val_items
    """

    labels = [y for _, y in items]

    train_items, val_items = train_test_split(
        items,
        test_size=0.2,
        stratify=labels,
        random_state=seed
    )

    train_ds = DigitsDataset(train_items, train_tfms)
    val_ds   = DigitsDataset(val_items, val_tfms)

    # Seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dl, val_dl, train_items, val_items


# ============================================================
# 5) MODEL
# ============================================================

def build_resnet18(num_classes: int = 10, freeze_backbone: bool = True):
    """
    Build a pretrained ResNet18 model for digit classification.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# 6) TRAINING
# ============================================================

def train(model, train_dl, val_dl, epochs: int, lr: float):
    """
    Train the model and keep the best validation model.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    @torch.no_grad()
    def eval_val_acc():
        model.eval()
        correct, total = 0, 0
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / total

    best_state = None
    best_acc = -1
    history = {"val_acc": []}

    for _ in range(epochs):
        model.train()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = eval_val_acc()
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, history


# ============================================================
# 7) EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataset_dir: str):
    """
    Evaluate a trained model on a given dataset directory.
    """

    items = collect_paths(dataset_dir)
    ds = DigitsDataset(items, val_tfms)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    y_true, y_pred = [], []

    model.eval()

    for x, y in dl:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

    acc = float(np.mean(np.array(y_pred) == np.array(y_true)))
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return acc, report, cm


# ============================================================
# 8) FULL PIPELINE ENTRY POINT
# ============================================================

def run_pipeline(
    train_dir: str = TRAIN_DIR,
    test_dir: str = TEST_DIR,
    seed: int = SEED,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
):
    """
    Full reproducible training + evaluation pipeline.
    """

    set_seed(seed)

    train_items = collect_paths(train_dir)
    train_dl, val_dl, _, _ = make_loaders(train_items, batch_size, seed)

    model = build_resnet18(10, freeze_backbone).to(device)

    model, history = train(model, train_dl, val_dl, epochs, lr)

    acc, report, cm = evaluate(model, test_dir)

    return model, history, acc, report, cm


# ============================================================
# 9) SCRIPT MODE (Optional)
# ============================================================

if __name__ == "__main__":
    model, hist, acc, report, cm = run_pipeline()
    print("Test Accuracy:", acc)
    print(report)
    print(cm)
