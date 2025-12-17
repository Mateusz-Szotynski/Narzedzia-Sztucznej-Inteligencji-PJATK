"""
Pima Indians Diabetes Dataset – Zbiór 1 (plik pimaIndiansDiabetes.csv z poprzednich ćwiczeń).

Autorzy: Robert Michałowski, Mateusz Szotyński


Opis problemu:
    Przewidywanie wystąpienia cukrzycy (0/1) na podstawie 8 cech
    medycznych (klasyfikacja binarna).

    W tym skrypcie realizujemy wymóg z kolejnego etapu:
    "Technologia dowolna: wybrać jeden framework wspierający użycie sieci neuronowych
     i użyć go do realizacji wszystkich ćwiczeń.
     Wykorzystać jeden z zbiorów danych z poprzednich ćwiczeń i nauczyć sieć neuronową.
     Porównać skuteczność obu podejść."

    Framework: PyTorch

Źródło opisu datasetu (MLMastery):
    https://machinelearningmastery.com/standard-machine-learning-datasets/

Uwagi dot. danych:
    W zbiorze Pima część wartości 0 w niektórych kolumnach medycznych
    może oznaczać brak danych (np. Glucose, BloodPressure, SkinThickness, Insulin, BMI).
    Dlatego w tym skrypcie:
      - zamieniamy 0 na NaN w wybranych kolumnach,
      - uzupełniamy braki medianą,
      - wykonujemy standaryzację cech (bardzo ważne dla sieci neuronowej),
      - uczymy prostą sieć neuronową MLP.

Instrukcja użycia:
    1. Upewnij się, że plik pimaIndiansDiabetes.csv istnieje w folderze:
       ../Zadania_Zjazd_4/pimaIndiansDiabetes.csv
       (czyli: w poprzednim folderze zjazdu 4)
    2. Przejdź do folderu: Zadania_zjazd_5
    3. Uruchom:
       python pytorchPimaNN.py

Wynik:
    Skrypt wypisze:
      - postęp treningu (loss),
      - metryki klasyfikacji (accuracy, precision, recall, f1-score),
      - macierz pomyłek,
      - przykładowe predykcje dla kilku próbek testowych.

    Output można skopiować do README lub zrobić zrzut ekranu i wrzucić do repozytorium.
"""

# biblioteki do obróbki danych
import numpy as np
import pandas as pd

# podział danych na trening/test
from sklearn.model_selection import train_test_split

# standaryzacja cech (ważne dla NN)
from sklearn.preprocessing import StandardScaler

# metryki oceny jakości
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Nazwy kolumn odpowiadające formatowi Pima Indians Diabetes (8 cech + Outcome)
COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",  # kolumna klasy 0/1
]


class DiabetesMLP(nn.Module):
    """
    Prosta sieć neuronowa typu MLP do klasyfikacji binarnej (0/1).

    Zwraca logit (bez sigmoid), ponieważ do straty używamy BCEWithLogitsLoss,
    które jest stabilne numerycznie.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # logit
        )

    def forward(self, x):
        return self.model(x)


def load_pima_csv(csv_path: str):
    """
    Wczytuje dane Pima z pliku pimaIndiansDiabetes.csv.

    Oczekiwany format:
        - brak nagłówka,
        - 9 kolumn: 8 cech numerycznych + 1 kolumna klasy (0/1).
    """
    # Wczytanie pliku CSV bez nagłówka i przypisanie własnych nazw kolumn
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

    # Kolumny, gdzie 0 może oznaczać brak danych (typowa praktyka dla Pima)
    cols_with_zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # Zamiana 0 na NaN w wybranych kolumnach
    for c in cols_with_zero_as_missing:
        df[c] = df[c].replace(0, np.nan)

    # Uzupełnienie braków medianą
    df = df.fillna(df.median(numeric_only=True))

    # X – cechy, y – etykieta
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)

    return X, y


def main():
    # 1. Wczytanie danych (korzystamy z tego samego zbioru co w klasycznych modelach)
    # Plik jest w folderze Zadania_Zjazd_4, a skrypt w Zadania_zjazd_5
    X, y = load_pima_csv("pimaIndiansDiabetes.csv")

    # 2. Podział na dane treningowe i testowe
    # stratify=y utrzymuje proporcje klas 0/1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Standaryzacja cech (ważne dla NN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 4. Przygotowanie danych dla PyTorch (tensory)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 5. Model + trening
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiabetesMLP(input_dim=X_train.shape[1]).to(device)

    # Strata i optymalizator
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 60

    # 6. Trening sieci neuronowej
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # forward: logity (bez sigmoid)
            logits = model(xb).squeeze(1)

            # obliczenie straty
            loss = criterion(logits, yb)

            # backprop i aktualizacja wag
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)

        # log co 10 epok (czytelne do screenshota)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | loss={epoch_loss:.4f}")

    # 7. Ewaluacja na zbiorze testowym
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device)).squeeze(1)
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        y_pred = (probs_test >= 0.5).astype(int)

    y_true = y_test.astype(int)

    print("\n=== ZBIÓR 1 – Neural Network (PyTorch MLP) ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # 8. Przykładowe wywołanie klasyfikatora dla kilku danych wejściowych
    print("\n=== Przykładowe predykcje (pierwsze 5 z testu) ===")
    for i in range(min(5, len(y_pred))):
        print(f"Próbka {i}: pred={y_pred[i]} | p(diabetes)={probs_test[i]:.3f} | true={y_true[i]}")


if __name__ == "__main__":
    main()
