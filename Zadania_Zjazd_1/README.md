\# Lights Out 3×3 — projekt NAI

\- Inspiracja: \*Lights Out\* (opis: https://en.wikipedia.org/wiki/Lights\_Out\_(game))



\## Autorzy

\- Robert Michałowski (s27137)

\- Mateusz Szotyński (s26963)



\## Zasady gry

Plansza 3×3. Ruch to wskazanie pola 0–8 (numeracja: `0 1 2 / 3 4 5 / 6 7 8`).

Kliknięte pole oraz sąsiedzi w góra/dół/lewo/prawo zmieniają stan (ON↔OFF).

Wygrywa gracz, który swoim ruchem zgasi wszystkie światła.



\- Deterministyczna, dwuosobowa, suma zerowa.

\- Warunek końca: stan `0` (wszystkie OFF) — zwycięża gracz, który wyłączy wszystkie światła.



\## Sztuczna inteligencja (adversarial search)

Zaimplementowano \*\*Negamax + alpha–beta\*\*, z:

\- detekcją cykli (remis przy powtórzeniu stanu w ścieżce),

\- prostą tablicą transpozycji (cache pozycji),

\- preferencją szybkich wygranych / opóźniania przegranej (użycie głębokości w ocenie terminali).



\## Wymagania

\- Python 3.10+

\- `easyAI`




## Uruchomienie



```bash

pip install easyAI

python Zadania\_Zjazd\_1/gra\_1.py

## Rozgrywka

Gęstość startowa (0..1, Enter=0.5):

=== Lights Out 3×3 (Negamax + Alpha–Beta) ===

Numeracja pól:

0 1 2

3 4 5

6 7 8

Człowiek zaczyna. Przeciwnik: player AI (Negamax + alpha–beta).



&nbsp;  c0 c1 c2     indeksy:

r0  · 1 ·     0 1 2

r1  · · ·     3 4 5

r2  1 1 ·     6 7 8

włączone: 3 | podaj numer pola (0–8)

Player 1 what do you play ? 7

Move #1: player 1 plays 7 :

...

✅ KONIEC: wygrał player AI (zgasił ostatnie światła).









