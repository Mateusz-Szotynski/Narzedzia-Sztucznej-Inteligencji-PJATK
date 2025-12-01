"""
Pima Indians Diabetes – plik diabetes.csv (zbiór 2).

Autorzy: Robert Michałowski, Mateusz Szotyński

Opis problemu:
    Przewidywanie wystąpienia cukrzycy w ciągu 5 lat na podstawie
    8 cech medycznych (Pima Indians Diabetes Dataset).

    Cechy:
        1. Number of times pregnant
        2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        3. Diastolic blood pressure (mm Hg)
        4. Triceps skinfold thickness (mm)
        5. 2-Hour serum insulin (mu U/ml)
        6. Body mass index (weight in kg/(height in m)^2)
        7. Diabetes pedigree function
        8. Age (years)
        9. Outcome (0 lub 1)

Instrukcja użycia:
    1. Umieść plik diabetes.csv w tym samym katalogu.
    2. Uruchom:
       python diabetes_full_classification.py

Przykładowe źródło danych online (link do wpisania w tabeli / komentarzu):
    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_diabetes(csv_path: str):
    """
    Wczytuje lokalny diabetes.csv.

    Zakładamy, że:
        - kolumna z etykietą nazywa się 'Outcome',
        - pozostałe kolumny to cechy numeryczne.
    """
    # Wczytanie pełnego pliku CSV do obiektu DataFrame
    df = pd.read_csv(csv_path)

    # X – wszystkie kolumny z cechami (bez kolumny Outcome)
    X = df.drop("Outcome", axis=1).values

    # y – wektor klas (0/1) z kolumny Outcome
    y = df["Outcome"].values
    return df, X, y


def visualize_pca(df: pd.DataFrame) -> None:
    """
    Przykładowa wizualizacja danych ze zbioru 2:
    PCA do 2 wymiarów + wykres punktowy klas 0/1.
    """
    # Oddzielenie cech od etykiet
    features = df.drop("Outcome", axis=1).values
    labels = df["Outcome"].values

    # PCA redukuje wymiar z 8 cech do 2 nowych wymiarów (PC1, PC2),
    # które są kombinacją wszystkich cech i najlepiej pokazują strukturę danych.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(features)

    # Rysowanie wykresu:
    # dla każdej klasy (0 i 1) zaznaczamy punkty innym kolorem.
    plt.figure(figsize=(6, 5))
    for cls, color, label in [(0, "tab:blue", "class 0"), (1, "tab:orange", "class 1")]:
        mask = labels == cls
        plt.scatter(
            X_pca[mask, 0],   # współrzędna PC1
            X_pca[mask, 1],   # współrzędna PC2
            c=color,
            label=label,
            alpha=0.7,
            edgecolor="k",
            s=40,
        )

    # Opisy wykresu
    plt.title("Diabetes (zbiór 2) – PCA 2D")
    plt.xlabel("PC1")   # pierwsza główna składowa (największa wariancja)
    plt.ylabel("PC2")   # druga główna składowa
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Trenuje drzewo decyzyjne dla zbioru diabetes.csv."""
    # max_depth=5 – ograniczamy głębokość drzewa
    # class_weight="balanced" – ważymy klasy odwrotnie do ich liczności
    # (bo chorych jest mniej niż zdrowych).
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_svms(X_train, y_train) -> Dict[str, SVC]:
    """
    Trenuje kilka wariantów SVM z różnymi kernelami i parametrami.

    Zwraca słownik: nazwa_modelu -> wytrenowany SVC.
    """
    # Lista konfiguracji (nazwa, parametry SVC).
    configs = [
        ("linear_C1", dict(kernel="linear", C=1.0)),
        ("linear_C10", dict(kernel="linear", C=10.0)),
        ("rbf_C1_gscale", dict(kernel="rbf", C=1.0, gamma="scale")),
        ("rbf_C10_g0.1", dict(kernel="rbf", C=10.0, gamma=0.1)),
        ("poly_deg3", dict(kernel="poly", degree=3, C=1.0, gamma="scale")),
    ]

    models: Dict[str, SVC] = {}
    for name, params in configs:
        # Tworzymy SVM z danym kernelem i parametrami
        clf = SVC(random_state=42, **params)
        # Uczymy model na zbiorze treningowym
        clf.fit(X_train, y_train)
        # Zapisujemy go pod nazwą, np. "rbf_C1_gscale"
        models[name] = clf
    return models


def evaluate_model(name: str, clf, X_test, y_test) -> float:
    """
    Wyświetla metryki jakości klasyfikacji i zwraca accuracy:

    - accuracy
    - classification_report (precision, recall, f1)
    - confusion_matrix
    """
    # Predykcje modelu dla zbioru testowego
    y_pred = clf.predict(X_test)

    # Accuracy – procent poprawnych predykcji
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)

    # Raport z precision, recall, f1-score i support dla każdej klasy
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Macierz pomyłek (TP, FP, FN, TN w formie tabeli)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Zwracamy accuracy, żeby później porównać modele między sobą
    return acc


def show_example_predictions(clf, X_test, y_test, n_examples: int = 5) -> None:
    """
    Wywołuje klasyfikator na kilku przykładowych danych wejściowych
    i wypisuje przewidywania vs rzeczywiste etykiety.
    """
    # Losujemy n_examples indeksów ze zbioru testowego (bez powtórzeń)
    idx = np.random.choice(len(X_test), size=n_examples, replace=False)

    # Pobieramy wylosowane próbki i ich prawdziwe etykiety
    X_samples = X_test[idx]
    y_true = y_test[idx]

    # Przewidywane klasy dla tych próbek
    y_pred = clf.predict(X_samples)

    print("\nPrzykładowe predykcje (zbiór 2):")
    for i in range(n_examples):
        # Dla każdej próbki pokazujemy: co przewidział model,
        # a jaka była rzeczywista klasa w danych.
        print(
            f"Przykład {i+1}: przewidywana klasa = {y_pred[i]}, "
            f"rzeczywista klasa = {y_true[i]}"
        )


def main():
    # 1. Wczytanie danych ze zbioru 2 (diabetes.csv)
    # df – pełny DataFrame (do wizualizacji),
    # X – macierz cech,
    # y – wektor klas (0/1).
    df, X, y = load_diabetes("diabetes.csv")

    # 2. Wizualizacja – PCA 2D
    # Redukujemy wymiar do 2 i rysujemy wykres klas 0/1.
    # Na wykresie każdy punkt to pacjent, kolor to klasa.
    visualize_pca(df)

    # 3. Podział na trening/test + skalowanie cech
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,       # 20% danych przeznaczamy na test
        random_state=42,
        stratify=y,          # zachowujemy proporcje klas 0/1
    )

    # Skalowanie cech
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Drzewo decyzyjne – trening i metryki
    tree = train_decision_tree(X_train, y_train)
    acc_tree = evaluate_model("Decision Tree (zbiór 2)", tree, X_test, y_test)

    # 5. SVM – różne kernelle i parametry
    # Trenujemy kilka wariantów SVM (linear, rbf, poly...), każdy z innymi
    # parametrami C, gamma, degree.
    svm_models = train_svms(X_train, y_train)
    svm_scores: Dict[str, float] = {}

    # Dla każdego modelu liczymy accuracy i zapisujemy je w słowniku
    for name, model in svm_models.items():
        acc = evaluate_model(f"SVM ({name}) – zbiór 2", model, X_test, y_test)
        svm_scores[name] = acc

    # 6. Najlepszy SVM – przykładowe predykcje
    # Wybieramy nazwę modelu z najwyższym accuracy
    best_name = max(svm_scores, key=svm_scores.get)
    best_model = svm_models[best_name]

    print(
        f"\nNajlepszy wariant SVM (zbiór 2): {best_name} "
        f"(accuracy={svm_scores[best_name]:.3f})"
    )

    # Na najlepszym modelu pokazujemy konkretne przykłady:
    # jaką klasę przewidział, a jaka była rzeczywista etykieta.
    show_example_predictions(best_model, X_test, y_test, n_examples=5)

    # 7. Podsumowanie wpływu kerneli – do raportu/README
    print("\nPodsumowanie wpływu kerneli SVM (zbiór 2, accuracy):")
    # Sortujemy modele od najlepszego do najgorszego po accuracy
    for name, acc in sorted(svm_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:15s} -> {acc:.3f}")


    print(
        "\nKomentarze:\n"
        "- kernelle liniowe ('linear') zakładają liniową granicę decyzyjną;\n"
        "- kernel RBF potrafi modelować nieliniowe zależności i zwykle daje\n"
        "  najlepsze wyniki przy dobrze dobranych parametrach C i gamma;\n"
        "- kernel wielomianowy ('poly') także modeluje nieliniowości, ale\n"
        "  przy domyślnych parametrach może generalizować gorzej niż RBF.\n"
    )


if __name__ == "__main__":
    # Punkt wejścia programu – po uruchomieniu pliku wykonywana jest funkcja main().
    main()
