"""
==============================================
  Movie Recommendation System with EM + Fuzzy Logic
==============================================

Autor: Mateusz Szotyński, Robert Michałowski 
Data: 16-11-2025  

Opis:
------
Skrypt implementuje kompletny silnik rekomendacji filmów na podstawie:
1. Grupowania użytkowników algorytmem EM (Gaussian Mixture)
2. Logiki rozmytej (fuzzy logic, scikit-fuzzy)
3. Danych o filmach pobieranych z OMDb API

Funkcje systemu:
----------------
Wczytywanie ocen użytkowników z pliku CSV  
Grupowanie użytkowników według podobieństw ocen  
Rekomendacje filmów (top 5), których użytkownik nie oglądał  
Antyrekomendacje (bottom 5), których użytkownik powinien unikać  
Pobieranie szczegółów filmów z API OMDb (opis, ocena IMDB, gatunek)

Instrukcja użycia:
------------------
1. Zainstaluj wymagane biblioteki:
   pip install pandas scikit-learn scikit-fuzzy rapidfuzz requests

2. Wprowadź swój OMDb API KEY w zmiennej:
   OMDB_API_KEY = "YOUR_KEY_HERE"

3. Umieść plik CSV w tym samym katalogu.

4. Uruchom:
   python program.py

Referencje:
-----------
- Gaussian Mixture Models (EM): https://scikit-learn.org/stable/modules/mixture.html
- Scikit-Fuzzy: https://pythonhosted.org/scikit-fuzzy/
- OMDb API: https://www.omdbapi.com/

==============================================
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
import requests


# -----------------------------------
# Konfiguracja API
# -----------------------------------
OMDB_API_KEY = "453cca62"


# =====================================================
# 1. Pobieranie informacji o filmie z OMDb
# =====================================================
def omdb_info(title: str) -> dict:
    """
    Pobiera szczegóły filmu z OMDb API.

    Parameters
    ----------
    title : str
        Tytuł filmu.

    Returns
    -------
    dict
        Informacje o filmie: tytuł, gatunek, rok, IMDB rating, opis.
    """
    params = {"t": title, "apikey": OMDB_API_KEY}
    try:
        r = requests.get("http://www.omdbapi.com/", params=params)
        data = r.json()
        return {
            "Title": data.get("Title", title),
            "Year": data.get("Year", "unknown"),
            "Genre": data.get("Genre", "unknown"),
            "IMDB": data.get("imdbRating", "N/A"),
            "Plot": data.get("Plot", "No description available")
        }
    except:
        return {"Title": title, "Year": "-", "Genre": "-", "IMDB": "-", "Plot": "-"}


# =====================================================
# 2. Logika rozmyta — ocena preferencji
# =====================================================
def fuzzy_preference(score: float) -> dict:
    """
    Przekształca ocenę filmu użytkownika na stopnie przynależności 
    do kategorii: low, medium, high.

    Parameters
    ----------
    score : float
        Średnia ocena filmu.

    Returns
    -------
    dict
        Stopnie przynależności fuzzy.
    """
    x = np.arange(0, 11, 1)
    low = fuzz.trimf(x, [0, 0, 4])
    medium = fuzz.trimf(x, [3, 5, 7])
    high = fuzz.trimf(x, [6, 10, 10])

    return {
        "low": fuzz.interp_membership(x, low, score),
        "medium": fuzz.interp_membership(x, medium, score),
        "high": fuzz.interp_membership(x, high, score)
    }


def fuzzy_weighted_score(score: float) -> float:
    """
    Oblicza końcowy fuzzy-score filmu na podstawie fuzzy logic.

    Returns
    -------
    float
        Wartość preferencji (0–1).
    """
    fp = fuzzy_preference(score)
    return fp["low"] * 0.2 + fp["medium"] * 0.5 + fp["high"] * 1.0


# =====================================================
# 3. Rekomendacje i antyrekomendacje
# =====================================================
def recommend(user: str, df: pd.DataFrame, pivot: pd.DataFrame) -> None:
    """
    Generuje rekomendacje i antyrekomendacje dla danego użytkownika.

    Parameters
    ----------
    user : str
        Nazwa użytkownika.
    df : DataFrame
        Dane w formacie: user, title, rating
    pivot : DataFrame
        Dane przekształcone na macierz user x movie.
    """
    if user not in pivot.index:
        print("Nie ma takiego użytkownika w bazie!")
        return

    user_cluster = pivot.loc[user, "cluster"]
    similar = pivot[pivot["cluster"] == user_cluster]

    # średnie oceny filmów w klastrze
    means = similar.drop(columns="cluster").mean()
    means = means.dropna()

    # filmy już oglądane
    seen = df[df["user"] == user]["title"].unique()

    # kandydaci do rekomendacji
    unseen = means[~means.index.isin(seen)]

    # wyliczenie fuzzy-score
    scores = {t: fuzzy_weighted_score(r) for t, r in unseen.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n================ Rekomendacje dla {user} ================")
    for title, score in ranked[:5]:
        info = omdb_info(title)
        print(f"\n🎬 {info['Title']} ({info['Year']})")
        print(f"Gatunek: {info['Genre']}")
        print(f"IMDB: {info['IMDB']}")
        print(f"Fuzzy-score: {score:.3f}")
        print(f"Opis: {info['Plot']}")

    # antyrekomendacje (najgorsze filmy)
    ranked_bad = sorted(scores.items(), key=lambda x: x[1])

    print(f"\n================ Anty-rekomendacje dla {user} ================")
    for title, score in ranked_bad[:5]:
        info = omdb_info(title)
        print(f"\n{info['Title']} ({info['Year']})")
        print(f"Gatunek: {info['Genre']}")
        print(f"IMDB: {info['IMDB']}")
        print(f"Fuzzy-score: {score:.3f}")
        print(f"Opis: {info['Plot']}")


# =====================================================
# GŁÓWNY PROGRAM
# =====================================================
if __name__ == "__main__":
    # Wczytanie danych
    df = pd.read_csv("dane.csv", header=None, names=["user", "title", "rating"])

    # Pivot user-movie
    pivot = df.pivot_table(index="user", columns="title", values="rating")

    # EM clustering
    gm = GaussianMixture(n_components=4, random_state=42)
    gm.fit(pivot.fillna(0))
    pivot["cluster"] = gm.predict(pivot.fillna(0))

    # Uruchom rekomendacje
    recommend("Paweł Czapiewski", df, pivot)
