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
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values
    return df, X, y


def visualize_pca(df: pd.DataFrame) -> None:
    """
    Przykładowa wizualizacja danych ze zbioru 2:
    PCA do 2 wymiarów + wykres punktowy klas 0/1.
    """
    features = df.drop("Outcome", axis=1).values
    labels = df["Outcome"].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(features)

    plt.figure(figsize=(6, 5))
    for cls, color, label in [(0, "tab:blue", "class 0"), (1, "tab:orange", "class 1")]:
        mask = labels == cls
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=label,
            alpha=0.7,
            edgecolor="k",
            s=40,
        )

    plt.title("Diabetes (zbiór 2) – PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Trenuje drzewo decyzyjne dla zbioru diabetes.csv."""
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_svms(X_train, y_train) -> Dict[str, SVC]:
    """
    Trenuje kilka wariantów SVM z różnymi kernelami i parametrami.

    Zwraca słownik: nazwa_modelu -> wytrenowany SVC.
    """
    configs = [
        ("linear_C1", dict(kernel="linear", C=1.0)),
        ("linear_C10", dict(kernel="linear", C=10.0)),
        ("rbf_C1_gscale", dict(kernel="rbf", C=1.0, gamma="scale")),
        ("rbf_C10_g0.1", dict(kernel="rbf", C=10.0, gamma=0.1)),
        ("poly_deg3", dict(kernel="poly", degree=3, C=1.0, gamma="scale")),
    ]

    models: Dict[str, SVC] = {}
    for name, params in configs:
        clf = SVC(random_state=42, **params)
        clf.fit(X_train, y_train)
        models[name] = clf
    return models


def evaluate_model(name: str, clf, X_test, y_test) -> float:
    """
    Wyświetla metryki jakości klasyfikacji i zwraca accuracy:

    - accuracy
    - classification_report (precision, recall, f1)
    - confusion_matrix
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc


def show_example_predictions(clf, X_test, y_test, n_examples: int = 5) -> None:
    """
    Wywołuje klasyfikator na kilku przykładowych danych wejściowych
    i wypisuje przewidywania vs rzeczywiste etykiety.
    """
    idx = np.random.choice(len(X_test), size=n_examples, replace=False)
    X_samples = X_test[idx]
    y_true = y_test[idx]
    y_pred = clf.predict(X_samples)

    print("\nPrzykładowe predykcje (zbiór 2):")
    for i in range(n_examples):
        print(
            f"Przykład {i+1}: przewidywana klasa = {y_pred[i]}, "
            f"rzeczywista klasa = {y_true[i]}"
        )


def main():
    # 1. Wczytanie danych ze zbioru 2
    df, X, y = load_diabetes("diabetes.csv")

    # 2. Wizualizacja – PCA 2D
    visualize_pca(df)

    # 3. Podział na trening/test + skalowanie cech
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Drzewo decyzyjne – metryki
    tree = train_decision_tree(X_train, y_train)
    acc_tree = evaluate_model("Decision Tree (zbiór 2)", tree, X_test, y_test)

    # 5. SVM – różne kernelle i parametry
    svm_models = train_svms(X_train, y_train)
    svm_scores: Dict[str, float] = {}
    for name, model in svm_models.items():
        acc = evaluate_model(f"SVM ({name}) – zbiór 2", model, X_test, y_test)
        svm_scores[name] = acc

    # 6. Najlepszy SVM – przykładowe predykcje
    best_name = max(svm_scores, key=svm_scores.get)
    best_model = svm_models[best_name]
    print(
        f"\nNajlepszy wariant SVM (zbiór 2): {best_name} "
        f"(accuracy={svm_scores[best_name]:.3f})"
    )
    show_example_predictions(best_model, X_test, y_test, n_examples=5)

    # 7. Krótkie podsumowanie wpływu kerneli – do użycia w raporcie/README
    print("\nPodsumowanie wpływu kerneli SVM (zbiór 2, accuracy):")
    for name, acc in sorted(svm_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:15s} -> {acc:.3f}")

    print(
        "\nKomentarz (możesz wkleić do dokumentacji):\n"
        "- kernelle liniowe ('linear') zakładają liniową granicę decyzyjną;\n"
        "- kernel RBF potrafi modelować nieliniowe zależności i zwykle daje\n"
        "  najlepsze wyniki przy dobrze dobranych parametrach C i gamma;\n"
        "- kernel wielomianowy ('poly') także modeluje nieliniowości, ale\n"
        "  przy domyślnych parametrach może generalizować gorzej niż RBF.\n"
    )


if __name__ == "__main__":
    main()
