"""
Autorzy: Mateusz Szotyński, Robert Michałowski
Projekt: Evil Ads Platform – trening modelu CNN

Opis problemu:
Celem programu jest wytrenowanie konwolucyjnej sieci neuronowej (CNN),
która rozpoznaje stan oka na obrazie:
- 0 – oko zamknięte
- 1 – oko otwarte

Model trenowany jest na zbiorze obrazów podzielonym
na dwie klasy (open / closed), a następnie zapisywany
do pliku eye_model.pth, który jest używany przez aplikację
ad_watcher.py w czasie rzeczywistym.

Struktura danych wejściowych (dataset):
dataset/
├── open/      # obrazy oczu otwartych
└── closed/    # obrazy oczu zamkniętych

Technologie:
- Python 3
- PyTorch
- torchvision
"""

# ================== IMPORTY PYTORCH ==================
import torch                          # główna biblioteka ML
import torch.nn as nn                 # warstwy sieci neuronowych
import torch.optim as optim           # optymalizatory
from torch.utils.data import DataLoader  # ładowanie danych

# ================== IMPORTY TORCHVISION ==================
from torchvision import datasets      # gotowe klasy datasetów
from torchvision import transforms    # transformacje obrazów

# ================== IMPORT MODELU ==================
from eye_model import EyeStateCNN     # własny model CNN

# ================== PARAMETRY TRENINGU ==================
BATCH_SIZE = 32       # liczba obrazów przetwarzanych jednocześnie
EPOCHS = 10           # liczba epok (pełnych przejść przez dataset)
LEARNING_RATE = 0.001 # krok uczenia optymalizatora

# ================== TRANSFORMACJE DANYCH ==================
# Transformacje wykonywane na każdym obrazie:
# 1. konwersja do skali szarości
# 2. zmiana rozmiaru do 64x64
# 3. konwersja do tensora
# 4. normalizacja wartości pikseli
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ================== WCZYTANIE DATASETU ==================
# ImageFolder automatycznie:
# - tworzy etykiety na podstawie nazw katalogów
# - open -> 1
# - closed -> 0
dataset = datasets.ImageFolder(
    root="../dataset",
    transform=transform
)

# ================== DATALOADER ==================
# DataLoader:
# - dzieli dane na batch'e
# - miesza dane (shuffle)
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ================== INICJALIZACJA MODELU ==================
model = EyeStateCNN()

# ================== FUNKCJA STRATY ==================
# CrossEntropyLoss:
# - standardowa funkcja straty dla klasyfikacji wieloklasowej
criterion = nn.CrossEntropyLoss()

# ================== OPTYMALIZATOR ==================
# Adam – popularny algorytm optymalizacji
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ================== PĘTLA TRENINGOWA ==================
for epoch in range(EPOCHS):
    running_loss = 0.0

    # Iteracja po batchach danych
    for images, labels in train_loader:
        # Zerowanie gradientów
        optimizer.zero_grad()

        # Przejście w przód (forward pass)
        outputs = model(images)

        # Obliczenie straty
        loss = criterion(outputs, labels)

        # Propagacja wsteczna (backpropagation)
        loss.backward()

        # Aktualizacja wag modelu
        optimizer.step()

        # Sumowanie straty
        running_loss += loss.item()

    # Informacja po każdej epoce
    print(
        f"Epoka [{epoch + 1}/{EPOCHS}], "
        f"Strata: {running_loss:.4f}"
    )

# ================== ZAPIS MODELU ==================
# Zapis wytrenowanych wag do pliku
torch.save(
    model.state_dict(),
    "eye_model.pth"
)

print("Model zapisany jako eye_model.pth")
