"""
Autorzy: Mateusz Szotyński, Robert Michałowski
Projekt: Evil Ads Platform

Opis problemu:
Program realizuje system monitorowania uwagi użytkownika
podczas oglądania reklamy wideo. Wykorzystuje kamerę
internetową oraz algorytmy widzenia komputerowego
i uczenia maszynowego.

Aplikacja analizuje strumień wideo w czasie rzeczywistym
i sprawdza, czy użytkownik ma otwarte oczy.
Jeżeli przez określony czas użytkownik nie patrzy na ekran,
reklama zostaje zatrzymana i wyświetlany jest komunikat ostrzegawczy.

Technologie:
- Python 3
- OpenCV (detekcja twarzy i oczu)
- PyTorch (klasyfikacja stanu oka)

Instrukcja użycia:
1. Podłącz kamerę internetową
2. Uruchom program poleceniem: python ad_watcher.py
3. Naciśnij ESC, aby zakończyć działanie programu
"""

# ================== BEZPIECZEŃSTWO WINDOWS ==================

import os
# Wymuszenie użycia CPU zamiast GPU.
# Zapobiega to problemom i crashom OpenCV/PyTorch na Windows.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings
# Wyłączenie ostrzeżeń (np. NumPy), aby nie zaśmiecać konsoli
warnings.filterwarnings("ignore")

# ================== IMPORTY ==================

import cv2      # OpenCV – obsługa kamery i obrazu
import torch    # PyTorch – obsługa modelu sieci neuronowej
import time     # Pomiar czasu (do wykrywania braku uwagi)

# Import własnego modelu CNN do klasyfikacji stanu oka
from model.eye_model import EyeStateCNN

# ================== KONFIGURACJA ==================

# Czas (w sekundach), po którym pojawi się alert,
# jeżeli użytkownik nie patrzy na ekran
ALERT_TIME = 5

# Indeks kamery:
# 0 – domyślna kamera
# 1 – alternatywna (często w laptopach)
CAMERA_INDEX = 1

# ================== MODEL ==================

# Utworzenie instancji modelu CNN
model = EyeStateCNN()

# Wczytanie wytrenowanych wag modelu z pliku
model.load_state_dict(
    torch.load("model/eye_model.pth", map_location="cpu")
)

# Przełączenie modelu w tryb ewaluacji (inference)
# Wyłącza dropout i uczenie
model.eval()

# ================== KLASYFIKATORY OPENCV ==================

# Klasyfikator Haar Cascade do detekcji twarzy
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Klasyfikator Haar Cascade do detekcji oczu
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ================== KAMERA (DIRECTSHOW) ==================

# Uruchomienie kamery z użyciem backendu DirectShow
# (najbardziej stabilny na Windows)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Sprawdzenie, czy kamera została poprawnie uruchomiona
if not cap.isOpened():
    print("NIE MOŻNA OTWORZYĆ KAMERY")
    print("Spróbuj zmienić CAMERA_INDEX na 1")
    exit(1)

print("Kamera uruchomiona")

# Zapamiętanie momentu ostatniego „patrzenia”
last_look_time = time.time()

# Flaga informująca, czy reklama jest zatrzymana
paused = False

# ================== FUNKCJE ==================

def preprocess_eye(img):
    """
    Funkcja przygotowuje obraz oka do wejścia modelu CNN.

    Parametry:
    img (numpy.ndarray) – fragment obrazu zawierający oko,
                          wycięty z klatki kamery

    Zwraca:
    torch.Tensor – tensor o rozmiarze (1, 1, 64, 64),
                   gotowy do użycia w modelu
    None – w przypadku błędu przetwarzania
    """
    try:
        # Konwersja obrazu do skali szarości
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Zmiana rozmiaru obrazu do 64x64 piksele
        img = cv2.resize(img, (64, 64))

        # Konwersja obrazu do tensora PyTorch
        tensor = torch.tensor(img, dtype=torch.float32)

        # Dodanie wymiarów:
        # - batch size
        # - liczba kanałów (1 – skala szarości)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Normalizacja wartości pikseli do zakresu [0,1]
        tensor = tensor / 255.0

        return tensor
    except Exception:
        # W przypadku błędu zwracamy None
        return None

# ================== PĘTLA GŁÓWNA ==================

while True:
    # Odczyt jednej klatki z kamery
    ret, frame = cap.read()

    # Jeśli nie udało się pobrać klatki – zakończ program
    if not ret:
        print("Nie można odczytać klatki z kamery")
        break

    # Konwersja całej klatki do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcja twarzy na obrazie
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Flaga informująca, czy użytkownik patrzy na ekran
    looking = False

    # Iteracja po wszystkich wykrytych twarzach
    for (x, y, w, h) in faces:
        # Wycięcie obszaru twarzy
        face = frame[y:y+h, x:x+w]

        # Detekcja oczu w obrębie twarzy
        eyes = eye_cascade.detectMultiScale(face)

        for (ex, ey, ew, eh) in eyes:
            # Wycięcie obrazu oka
            eye_img = face[ey:ey+eh, ex:ex+ew]

            # Przygotowanie obrazu do modelu
            inp = preprocess_eye(eye_img)
            if inp is None:
                continue

            # Predykcja stanu oka bez obliczania gradientów
            with torch.no_grad():
                pred = torch.argmax(model(inp)).item()

            # Klasa 1 oznacza oko otwarte
            if pred == 1:
                looking = True

            # Kolor ramki:
            # zielony – oko otwarte
            # czerwony – oko zamknięte
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            # Rysowanie prostokąta wokół oka
            cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), color, 2)

    # ================== LOGIKA REKLAMY ==================

    if looking:
        # Jeśli użytkownik patrzy – resetujemy licznik czasu
        last_look_time = time.time()
        paused = False
    else:
        # Jeśli nie patrzy – sprawdzamy czas
        if time.time() - last_look_time > ALERT_TIME:
            paused = True
            cv2.putText(
                frame,
                "WROC DO OGLADANIA REKLAMY!",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    # Tekst informujący o stanie reklamy
    status = "REKLAMA ODTWARZANA" if not paused else "REKLAMA ZATRZYMANA"
    color = (0, 255, 0) if not paused else (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Wyświetlenie obrazu w oknie
    cv2.imshow("Evil Ads Platform", frame)

    # Klawisz ESC kończy program
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== SPRZĄTANIE ==================

# Zwolnienie kamery
cap.release()

# Zamknięcie wszystkich okien OpenCV
cv2.destroyAllWindows()

print("Program zakończony")
