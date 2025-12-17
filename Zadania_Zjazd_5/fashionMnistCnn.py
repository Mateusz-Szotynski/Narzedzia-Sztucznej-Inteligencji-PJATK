"""
Fashion-MNIST – klasyfikacja ubrań (PyTorch).

Autorzy: Robert Michałowski, Mateusz Szotyński

Opis problemu:
    Klasyfikacja obrazów ubrań zapisanych jako obrazy 28x28 pikseli
    w skali szarości. Każdy obraz należy do jednej z 10 klas
    (np. spodnie, koszulki, buty).

Źródło danych:
    Fashion-MNIST (Zalando Research)
    https://github.com/zalandoresearch/fashion-mnist

Cel skryptu:
    - nauczyć konwolucyjną sieć neuronową (CNN) rozpoznawać ubrania,
    - uzyskać dokładność (accuracy) na zbiorze testowym,
    - wygenerować confusion matrix jako plik PNG.

Instrukcja uruchomienia:
    python fashion_mnist_cnn.py --model small
    python fashion_mnist_cnn.py --model large
"""

# =========================
# IMPORTY BIBLIOTEK
# =========================

# Argumenty linii poleceń (np. wybór rozmiaru sieci)
import argparse

# Operacje numeryczne
import numpy as np

# PyTorch – główna biblioteka do sieci neuronowych
import torch
import torch.nn as nn

# DataLoader – ładowanie danych w mini-batchach
from torch.utils.data import DataLoader

# torchvision – gotowe zbiory danych i transformacje obrazów
import torchvision
import torchvision.transforms as transforms

# Metryki do oceny klasyfikacji
from sklearn.metrics import accuracy_score, confusion_matrix

# Biblioteka do rysowania wykresów (confusion matrix)
import matplotlib.pyplot as plt


# =========================
# NAZWY KLAS (etykiety)
# =========================

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================
# DEFINICJE MODELI CNN
# =========================

class SmallCNN(nn.Module):
    """
    Mniejsza sieć konwolucyjna (CNN).

    - mniej warstw
    - mniej parametrów
    - szybszy trening
    - zwykle trochę gorsza jakość niż większy model
    """
    def __init__(self):
        super().__init__()

        # Część konwolucyjna – ekstrakcja cech z obrazu
        self.features = nn.Sequential(
            # 1 kanał wejściowy (obraz grayscale)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),                 # funkcja aktywacji
            nn.MaxPool2d(2),           # zmniejszenie rozmiaru (28 -> 14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # (14 -> 7)
        )

        # Część klasyfikująca
        self.classifier = nn.Sequential(
            nn.Flatten(),              # spłaszczenie map cech do wektora
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),           # regularizacja – zmniejsza overfitting
            nn.Linear(128, 10)          # 10 klas wyjściowych
        )

    def forward(self, x):
        """
        Definiuje przepływ danych przez sieć (forward pass).
        """
        x = self.features(x)
        return self.classifier(x)


class LargeCNN(nn.Module):
    """
    Większa sieć CNN.

    - więcej warstw
    - więcej parametrów
    - wolniejsza
    - zwykle lepsza dokładność
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# CONFUSION MATRIX
# =========================

def plot_confusion_matrix(cm, labels, title, out_path):
    """
    Rysuje i zapisuje macierz pomyłek do pliku PNG.

    cm      – macierz pomyłek
    labels – nazwy klas
    """
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# FUNKCJA GŁÓWNA
# =========================

def main():
    # Argumenty uruchomieniowe
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "large"], default="small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Sprawdzenie czy mamy GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformacje obrazów:
    # - konwersja do tensora
    # - normalizacja wartości pikseli
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Automatyczne pobranie zbioru Fashion-MNIST
    train_set = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # DataLoader – porcjowanie danych (mini-batche)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Wybór architektury sieci
    model = SmallCNN() if args.model == "small" else LargeCNN()
    model.to(device)

    # Funkcja straty – klasyfikacja wieloklasowa
    criterion = nn.CrossEntropyLoss()

    # Optymalizator – aktualizacja wag
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # =========================
    # TRENING
    # =========================
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()       # zerowanie gradientów
            logits = model(xb)          # forward
            loss = criterion(logits, yb)
            loss.backward()             # backpropagation
            optimizer.step()            # aktualizacja wag

    # =========================
    # EWALUACJA
    # =========================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = torch.argmax(model(xb), dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())

    # Accuracy – główna miara
    acc = accuracy_score(y_true, y_pred)
    print("Fashion-MNIST Accuracy:", acc)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    out_file = f"confusion_matrix_fashion_{args.model}.png"
    plot_confusion_matrix(cm, CLASS_NAMES, "Confusion Matrix – Fashion MNIST", out_file)

    print("Confusion matrix saved to:", out_file)


if __name__ == "__main__":
    main()
