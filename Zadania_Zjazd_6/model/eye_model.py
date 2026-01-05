"""
Autorzy: Mateusz Szotyński, Robert Michałowski
Opis: Prosty model konwolucyjnej sieci neuronowej (CNN) do klasyfikacji stanu oka.

Klasy:
0 – oko zamknięte
1 – oko otwarte
"""

import torch.nn as nn
import torch.nn.functional as F


class EyeStateCNN(nn.Module):
    """
    Konwolucyjna sieć neuronowa do klasyfikacji obrazów oczu.
    """

    def __init__(self):
        """
        Konstruktor modelu.
        Definiuje warstwy sieci.
        """
        super().__init__()

        # Warstwa konwolucyjna nr 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Warstwa konwolucyjna nr 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Warstwa pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Warstwa w pełni połączona
        self.fc1 = nn.Linear(32 * 16 * 16, 64)

        # Warstwa wyjściowa (2 klasy)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Przejście w przód przez sieć.

        Parametry:
        x (torch.Tensor) – tensor wejściowy (batch, 1, 64, 64)

        Zwraca:
        torch.Tensor – logity klas
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Spłaszczenie tensora
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
