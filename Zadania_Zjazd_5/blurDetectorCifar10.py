"""
ZASKOCZ MNIE – klasyfikacja jakości obrazu: "ostry" vs "rozmyty" (PyTorch).

Autorzy: Robert Michałowski, Mateusz Szotyński

Opis problemu:
    Celem jest nauczenie sieci neuronowej rozpoznawania, czy obraz jest:
      - ostry (sharp)
      - rozmyty (blurry)

    Jest to praktyczny problem spotykany m.in. w:
      - kontroli jakości zdjęć (np. w aplikacjach mobilnych),
      - wykrywaniu poruszenia aparatu,
      - selekcji danych przed dalszą analizą (OCR, rozpoznawanie obiektów).

Dane:
    Wykorzystujemy obrazy CIFAR-10 (32x32 RGB), ale:
      - NIE używamy oryginalnych etykiet klas,
      - etykiety tworzymy samodzielnie:
          0 – obraz ostry (bez modyfikacji),
          1 – obraz rozmyty (Gaussian Blur).

Źródło danych:
    CIFAR-10:
    https://www.cs.toronto.edu/~kriz/cifar.html
"""

# ============================================================
# IMPORTY
# ============================================================

# Losowość – potrzebna do losowego rozmywania obrazów
import random

# Główna biblioteka do sieci neuronowych
import torch
import torch.nn as nn

# Dataset i DataLoader – obsługa danych i batchy
from torch.utils.data import Dataset, DataLoader

# torchvision – gotowe zbiory danych i transformacje obrazów
import torchvision
import torchvision.transforms as transforms


# ============================================================
# DATASET: CIFAR-10 → SHARP / BLURRY
# ============================================================

class CifarSharpBlurry(Dataset):
    """
    Własny dataset binarny oparty o CIFAR-10.

    Każdy obraz jest losowo:
      - pozostawiany bez zmian → label = 0 (sharp),
      - rozmywany filtrem Gaussa → label = 1 (blurry).

    Dzięki temu:
      - nie potrzebujemy ręcznie etykietować danych,
      - możemy generować duży zbiór uczący automatycznie.
    """

    def __init__(self, base_dataset, blur_prob=0.5):
        """
        base_dataset – oryginalny CIFAR-10
        blur_prob    – prawdopodobieństwo rozmycia obrazu
        """
        self.base = base_dataset
        self.blur_prob = blur_prob

        # Transformacja GaussianBlur – symuluje poruszenie aparatu
        # kernel_size: rozmiar filtra
        # sigma: zakres siły rozmycia
        self.blur = transforms.GaussianBlur(
            kernel_size=5,
            sigma=(1.0, 2.0)
        )

    def __len__(self):
        # Dataset ma tyle próbek, ile CIFAR-10
        return len(self.base)

    def __getitem__(self, idx):
        """
        Zwraca pojedynczą próbkę:
          - obraz (tensor)
          - etykieta: 0 = sharp, 1 = blurry
        """
        img, _ = self.base[idx]  # oryginalna etykieta CIFAR-10 jest ignorowana

        # Losowa decyzja: czy rozmyć obraz
        if random.random() < self.blur_prob:
            img = self.blur(img)
            label = 1  # blurry
        else:
            label = 0  # sharp

        return img, label


# ============================================================
# MODEL CNN – KLASYFIKACJA BINARNA
# ============================================================

class BlurCNN(nn.Module):
    """
    Prosta konwolucyjna sieć neuronowa (CNN)
    do klasyfikacji binarnej: sharp vs blurry.
    """

    def __init__(self):
        super().__init__()

        # Część konwolucyjna – ekstrakcja cech z obrazu
        self.features = nn.Sequential(
            # Wejście: obraz RGB (3 kanały)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 → 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 → 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 → 4x4
        )

        # Część klasyfikująca
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # mapy cech → wektor
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),                # ograniczenie overfittingu
            nn.Linear(128, 2)               # 2 klasy: sharp / blurry
        )

    def forward(self, x):
        """
        Przepływ danych przez sieć (forward pass).
        """
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# FUNKCJE TRENINGU I EWALUACJI
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Uczy model przez jedną epokę.

    Zwraca:
      - średnią stratę (loss),
      - accuracy na zbiorze treningowym.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()           # zerowanie gradientów
        logits = model(xb)              # forward
        loss = criterion(logits, yb)    # obliczenie straty
        loss.backward()                 # backpropagation
        optimizer.step()                # aktualizacja wag

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Ewaluacja modelu (bez uczenia).
    Zwraca accuracy.
    """
    model.eval()

    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = torch.argmax(model(xb), dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / total


# ============================================================
# MAIN
# ============================================================

def main():
    # Ustalenie ziarna losowości – powtarzalne wyniki
    random.seed(42)
    torch.manual_seed(42)

    # Wybór urządzenia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Parametry treningu
    epochs = 5
    batch_size = 128
    lr = 1e-3

    # Podstawowe transformacje CIFAR-10:
    # - konwersja do tensora
    # - normalizacja pikseli
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    # Wczytanie CIFAR-10 (oryginalne etykiety nie są używane)
    cifar_train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=base_transform
    )

    cifar_test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=base_transform
    )

    # Utworzenie datasetów sharp/blurry
    train_set = CifarSharpBlurry(cifar_train, blur_prob=0.5)
    test_set = CifarSharpBlurry(cifar_test, blur_prob=0.5)

    # DataLoadery – batchowanie danych
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model + funkcja straty + optymalizator
    model = BlurCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trening z logami (idealne do screenshota)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    # Wynik końcowy
    final_acc = evaluate(model, test_loader, device)
    print("\n=== WYNIK KOŃCOWY ===")
    print("Test Accuracy:", final_acc)
    print("Klasy: 0 = sharp (ostry), 1 = blurry (rozmyty)")


if __name__ == "__main__":
    main()
