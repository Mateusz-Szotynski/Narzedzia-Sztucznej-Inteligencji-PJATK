#
# # Lights Out 3×3 — projekt NAI
# - Inspiracja: *Lights Out* (opis: https://en.wikipedia.org/wiki/Lights_Out_(game))
#
# ## Autorzy
# - Robert Michałowski (s27137)
# - Mateusz Szotyński (s26963)
#
# ## Zasady gry
# Plansza 3×3. Ruch to wskazanie pola 0–8 (numeracja: `0 1 2 / 3 4 5 / 6 7 8`).
# Kliknięte pole oraz sąsiedzi w góra/dół/lewo/prawo zmieniają stan (ON↔OFF).
# Wygrywa gracz, który swoim ruchem zgasi wszystkie światła.
#
# - Deterministyczna, dwuosobowa, suma zerowa.
# - Warunek końca: stan `0` (wszystkie OFF) — zwycięża gracz, który wyłączy wszystkie światła.
#
#
# ## Sztuczna inteligencja (adversarial search)
# Zaimplementowano **Negamax + alpha–beta**, z:
# - detekcją cykli (remis przy powtórzeniu stanu w ścieżce),
# - prostą tablicą transpozycji (cache pozycji),
# - preferencją szybkich wygranych / opóźniania przegranej (użycie głębokości w ocenie terminali).
#
# ## Wymagania
# - Python 3.10+
# - `easyAI`
#
# ```
# pip install -r easyAI
# ```
#
# ## Uruchomienie
# ```
# python gra.py
# ```
# Program zapyta o gęstość startową świateł (0..1). Gracz 1 (Human) zawsze zaczyna, gracz 2 to **player AI**.
#
# ## Przykładowa rozgrywka (fragment)
# ```
# Gęstość startowa (0..1, Enter=0.5):
# === Lights Out 3×3 (Negamax + Alpha–Beta) ===
# Numeracja pól:
# 0 1 2
# 3 4 5
# 6 7 8
# Człowiek zaczyna. Przeciwnik: player AI (Negamax + alpha–beta).
#
#    c0 c1 c2     indeksy:
# r0  · 1 ·     0 1 2
# r1  · · ·     3 4 5
# r2  1 1 ·     6 7 8
# włączone: 3 | podaj numer pola (0–8)
# Player 1 what do you play ? 7
# Move #1: player 1 plays 7 :
# ...
# ✅ KONIEC: wygrał player AI (zgasił ostatnie światła).
# ```





from __future__ import annotations
import random
from easyAI import TwoPlayerGame, Human_Player, AI_Player

# --- maski dla 3x3 (pole + N,S,E,W) ---
MASKS = []
for r in range(3):
    for c in range(3):
        m = 0
        for rr, cc in ((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < 3 and 0 <= cc < 3:
                m |= 1 << (rr * 3 + cc)
        MASKS.append(m)

def toggle(state: int, r: int, c: int) -> int:
    return state ^ MASKS[r * 3 + c]


# Przełącza światło na polu (r, c) oraz jego sąsiadów
#     (góra, dół, lewo, prawo).
#
#     Parameters:
#         state (int): stan planszy jako 9-bitowa liczba binarna
#         r (int): wiersz (0–2)
#         c (int): kolumna (0–2)
#
#     Returns:
#         int: nowy stan planszy po przełączeniu pól


def rand_state(density: float = 0.5) -> int:

    # Generuje losowy stan planszy 3x3. `density` to
    # prawdopodobieństwo, że dane pole jest włączone (1).
    # Zwraca stan jako liczbę całkowitą (9-bitową).

    while True:
        s = sum(1 << i for i in range(9) if random.random() < density)
        if s != 0:
            return s

# ------------------- Negamax + Alpha–Beta -------------------

WIN_SCORE = 10_000
class ABNegamaxAI:

    # Sztuczna inteligencja używająca Negamax z przycinaniem alpha–beta.
    # Używa detekcji cykli (remis przy powtórzeniu stanu w ścieżce)
    # oraz prostej tablicy transpozycji (cache pozycji).
    # Preferuje szybkie wygrane i opóźnianie przegranej


    def __init__(self):
        # cache pozycji (transposition table)
        # Inicjalizuje pamięć transpozycji (cache).

        self.cache = {}


    def __call__(self, game: "LightsOut3x3") -> str:
        # Zwraca najlepszy ruch dla danego stanu gry.
        # Parameters:
        #     game (LightsOut3x3): aktualny stan gry
        # Returns:
        #     str: najlepszy ruch jako indeks pola (0–8).

        state = game.state
        score, move = self._negamax(state, -WIN_SCORE, WIN_SCORE, 0, set())
        return str(move or 0)

    def _negamax(self, state: int, alpha: int, beta: int, depth: int, path: set[int]) -> tuple[int, int | None]:

        # Negamax z przycinaniem alpha–beta, detekcją cykli i cache.
        # Parameters:
        #     state (int): aktualny stan planszy jako 9-bitowa liczba binarna
        #     alpha (int): wartość alpha dla przycinania
        #     beta (int): wartość beta dla przycinania
        #     depth (int): głębokość w drzewie przeszukiwania
        #     path (set[int]): zbiór stanów w bieżącej ścieżce (dla detekcji cykli)
        # Returns:
        #     tuple[int, int | None]: (ocena stanu, najlepszy ruch jako indeks pola lub None)

        if state == 0:
            return -WIN_SCORE + depth, None
        if state in path:
            return 0, None
        if state in self.cache:
            return self.cache[state]

        best_score, best_move = -WIN_SCORE * 2, None
        path.add(state)
        for i in range(9):
            r, c = divmod(i, 3)
            nxt = toggle(state, r, c)
            score, _ = self._negamax(nxt, -beta, -alpha, depth + 1, path)
            score = -score
            if score > best_score:
                best_score, best_move = score, i
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        path.remove(state)
        self.cache[state] = (best_score, best_move)
        return best_score, best_move

# ------------------- Gra easyAI -------------------

class LightsOut3x3(TwoPlayerGame):

    # Gra Lights Out 3x3 dla easyAI.
    # Stan planszy jest reprezentowany jako liczba całkowita (9-bitowa).


    def __init__(self, players, start=None):

        # Inicjalizuje grę z podanymi graczami i opcjonalnym stanem startowym.
        # Parameters:
        #     players (list): lista graczy (Human_Player lub AI_Player)
        #     start (int | None): opcjonalny stan startowy jako 9-bitowa liczba binarna
        #                         (jeśli None, generowany losowo z gęstością 0.5)


        self.players = players
        self.state = start or rand_state(0.5)
        self.current_player = 1
        self.move_count = 0

    def possible_moves(self): return [str(i) for i in range(9)]

    # Zwraca listę możliwych ruchów jako indeksy pól (0–8).

    def make_move(self, move):

        # Wykonuje ruch, przełączając światła na wskazanym polu i jego sąsiadach.

        i = int(move)
        r, c = divmod(i, 3)
        self.state = toggle(self.state, r, c)
        self.move_count += 1
    def win(self): return self.state == 0

    # Sprawdza, czy gra została wygrana (wszystkie światła wyłączone).

    def is_over(self): return self.win() or self.move_count > 200

    # Sprawdza, czy gra się zakończyła (wygrana lub przekroczono limit ruchów).

    def show(self):

        # Wyświetla aktualny stan planszy w czytelnej formie.

        print("\n   c0 c1 c2     indeksy:")
        for r in range(3):
            vals = [("1" if (self.state >> (r*3+c)) & 1 else "·") for c in range(3)]
            idxs = [str(r*3+c) for c in range(3)]
            print(f"r{r}  {' '.join(vals)}     {' '.join(idxs)}")
        print(f"włączone: {self.state.bit_count()} | podaj numer pola (0–8)")

# ------------------- Main -------------------

if __name__ == "__main__":

    # Ustawienia początkowe

    random.seed()
    try:
        dens = float(input("Gęstość startowa (0..1, Enter=0.5): ") or 0.5)
    except Exception:
        dens = 0.5
    start = rand_state(dens)
    ai = ABNegamaxAI()

    # auto-wybór kolejności: AI zaczyna tylko jeśli ma wygrywającą pozycję
    score, _ = ai._negamax(start, -WIN_SCORE, WIN_SCORE, 0, set())
    if score > 0:
        players = [AI_Player(ai), Human_Player()]
        order_info = "AI zaczyna (ma wygrywającą pozycję)."
    else:
        players = [Human_Player(), AI_Player(ai)]
        order_info = "Człowiek zaczyna."

    # podpisy graczy
    players[0].name = "player 1 (Human)"
    players[1].name = "player AI"

    print("=== Lights Out 3×3 (Negamax + Alpha–Beta) ===")
    print(order_info)
    print("Numeracja pól:\n0 1 2\n3 4 5\n6 7 8")

    game = LightsOut3x3(players=players, start=start)
    history = game.play()

    # komunikat końcowy
    if game.state == 0:
        winner = 2 if game.current_player == 1 else 1
        print(f"\n✅ KONIEC: wygrał player {winner} ({players[winner-1].name}).")
    elif game.move_count > 200:
        print("\n⏸️  REMIS: przekroczono limit ruchów.")
    else:
        print("\nKONIEC: gra przerwana.")
