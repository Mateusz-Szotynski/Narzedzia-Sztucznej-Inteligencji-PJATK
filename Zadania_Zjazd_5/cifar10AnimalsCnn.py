"""
CIFAR-10 – klasyfikacja obrazów (zwierzęta) z porównaniem dwóch rozmiarów sieci CNN.

Autorzy: Robert Michałowski, Mateusz Szotyński

Opis problemu:
    Zbiór CIFAR-10 zawiera 10 klas obrazów 32x32 (RGB), w tym 6 klas zwierząt:
    bird, cat, deer, dog, frog, horse.

Cel skryptu:
    - nauczyć sieć neuronową rozpoznawać zwierzęta (CIFAR-10),
    - porównać dwie architektury CNN: SMALL i LARGE,
    - wygenerować logi treningu i wyniki końcowe do repozytorium,

Źródło danych:
    https://www.cs.toronto.edu/~kriz/cifar.html

Instrukcja uruchomienia:
    python cifar10_animals_cnn_compare.py
"""

# =========================
# IMPORTY
# =========================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


# =========================
# NAZWY KLAS
# =========================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

ANIMAL_CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]


# =========================
# MODELE CNN
# =========================

class SmallCIFARCNN(nn.Module):
    """Mniejsza, szybsza sieć CNN."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LargeCIFARCNN(nn.Module):
    """Większa, dokładniejsza sieć CNN."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# FUNKCJE TRENING / TEST
# =========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = torch.argmax(model(xb), dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / total


# =========================
# MAIN – AUTOMATYCZNE PORÓWNANIE
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Animal classes:", ANIMAL_CLASSES)

    epochs = 10
    batch_size = 128
    lr = 1e-3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    results = {}

    for name, model_class in [("SMALL", SmallCIFARCNN), ("LARGE", LargeCIFARCNN)]:
        print(f"\n===== TRAINING MODEL: {name} =====")

        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_acc = evaluate(model, test_loader, device)

            print(f"{name} | Epoch {epoch:02d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"train_acc={train_acc:.4f} | "
                  f"test_acc={test_acc:.4f}")

        final_acc = evaluate(model, test_loader, device)
        results[name] = final_acc

    # =========================
    # PODSUMOWANIE
    # =========================

    print("\n===== PORÓWNANIE MODELI =====")
    for name, acc in results.items():
        print(f"{name} Test Accuracy: {acc:.4f}")

    print("\nWniosek:")
    print("Model LARGE osiąga zwykle wyższą dokładność kosztem dłuższego treningu.")


if __name__ == "__main__":
    main()
