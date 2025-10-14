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

def rand_state(density: float = 0.5) -> int:
    while True:
        s = sum(1 << i for i in range(9) if random.random() < density)
        if s != 0:
            return s

# ------------------- Negamax + Alpha–Beta -------------------

WIN_SCORE = 10_000

class ABNegamaxAI:
    def __init__(self):
        self.cache = {}

    def __call__(self, game: "LightsOut3x3") -> str:
        state = game.state
        score, move = self._negamax(state, -WIN_SCORE, WIN_SCORE, 0, set())
        return str(move or 0)

    def _negamax(self, state: int, alpha: int, beta: int, depth: int, path: set[int]) -> tuple[int, int | None]:
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
    def __init__(self, players, start=None):
        self.players = players
        self.state = start or rand_state(0.5)
        self.current_player = 1
        self.move_count = 0

    def possible_moves(self): return [str(i) for i in range(9)]
    def make_move(self, move):
        i = int(move)
        r, c = divmod(i, 3)
        self.state = toggle(self.state, r, c)
        self.move_count += 1
    def win(self): return self.state == 0
    def is_over(self): return self.win() or self.move_count > 200
    def show(self):
        print("\n   c0 c1 c2     indeksy:")
        for r in range(3):
            vals = [("1" if (self.state >> (r*3+c)) & 1 else "·") for c in range(3)]
            idxs = [str(r*3+c) for c in range(3)]
            print(f"r{r}  {' '.join(vals)}     {' '.join(idxs)}")
        print(f"włączone: {self.state.bit_count()} | podaj numer pola (0–8)")

# ------------------- Main -------------------

if __name__ == "__main__":
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
