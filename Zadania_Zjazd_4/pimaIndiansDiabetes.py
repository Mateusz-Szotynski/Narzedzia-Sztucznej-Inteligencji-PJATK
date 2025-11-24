"""
Pima Indians Diabetes Dataset – Zbiór 1 (dane z MLmastery, plik pimaIndiansDiabetes.csv).

Autorzy: Robert Michałowski, Mateusz Szotyński


Opis problemu:
    Przewidywanie wystąpienia cukrzycy (0/1) na podstawie 8 cech
    medycznych, zgodnie z opisem na stronie:

    https://machinelearningmastery.com/standard-machine-learning-datasets/

    W tym skrypcie dla Zbioru 1 realizujemy wymóg:
    "Naucz drzewo decyzyjne i SVM klasyfikować dane."

Instrukcja użycia:
    1. Umieść plik pimaIndiansDiabetes.csv w tym samym katalogu co skrypt.
       Format: 8 cech numerycznych + 1 kolumna klasy (0/1), bez nagłówka.
    2. Uruchom:
       python pima_web_basic.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Nazwy kolumn odpowiadające formatowi Pima Indians Diabetes
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


def load_pima_web(csv_path: str):
    """
    Wczytuje dane Pima ze Zbioru 1 z pliku pimaIndiansDiabetes.csv.

    Oczekiwany format:
        - brak nagłówka,
        - 9 kolumn: 8 cech numerycznych + 1 kolumna klasy (0/1).
    """
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
    X = df.iloc[:, :-1].values
    y = df["Outcome"].values
    return X, y


def main():
    # 1. Wczytanie danych ze Zbioru 1
    X, y = load_pima_web("pimaIndiansDiabetes.csv")

    # 2. Podział na trening/test + skalowanie (szczególnie ważne dla SVM)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Drzewo decyzyjne – uczymy klasyfikować dane
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_tree = tree.predict(X_test)
    print("=== ZBIÓR 1 – Decision Tree ===")
    print("Accuracy:", accuracy_score(y_test, y_tree))
    print(classification_report(y_test, y_tree))

    # 4. SVM – uczymy klasyfikować dane (jeden wariant, np. RBF)
    svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_clf.fit(X_train, y_train)
    y_svm = svm_clf.predict(X_test)
    print("=== ZBIÓR 1 – SVM (RBF) ===")
    print("Accuracy:", accuracy_score(y_test, y_svm))
    print(classification_report(y_test, y_svm))


if __name__ == "__main__":
    main()
