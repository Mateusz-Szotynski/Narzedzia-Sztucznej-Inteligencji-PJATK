"""
===============================================================
PROBLEM: SYSTEM HAMOWANIA OPARTY NA LOGICE ROZMYTEJ
---------------------------------------------------------------
Opis:
Program symuluje działanie inteligentnego systemu hamowania
z wykorzystaniem logiki rozmytej (fuzzy logic).

System na podstawie trzech danych wejściowych:
 - prędkości pojazdu (km/h),
 - odległości od przeszkody (m),
 - warunków na drodze (sucha, wilgotna, mokra),
określa rekomendowaną siłę hamowania w procentach (0–100%).

Wynik działania programu to "procent siły hamowania", który
odpowiada wartości nacisku na pedał hamulca – im wyższy procent,
tym silniejsze hamowanie.

---------------------------------------------------------------
Autorzy rozwiązania:
 - Mateusz Szotyński
 - Robert Michałowski

Data: Listopad 2025
Język: Python 3.11+
===============================================================

INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA:
1: Upewnij się, że masz zainstalowanego Pythona (>=3.10)
2: Zainstaluj wymagane biblioteki:
     pip install numpy scikit-fuzzy matplotlib
3:  Zapisz ten plik jako: fuzzy_braking_system.py
4: Uruchom w terminalu:
     python fuzzy_braking_system.py
===============================================================
"""
import numpy as np # For numerical arrays and ranges
import skfuzzy as fuzz # For fuzzy logic operations (membership functions)
from skfuzzy import control as ctrl # For fuzzy control systems (rules, simulation)

"""Define fuzzy variables."""
'''ctrl.Antecedent creates universe of a given name.
Whereas np.arange returns an array of evenly spaced values (from 0 to 120 include, spaced by 1)'''
# Define input variable: speed (range 0-120 km/h)
speed = ctrl.Antecedent(np.arange(0, 121, 1), 'speed')

# Define input variable: distance (range 0-100 meters)
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')

# Define input variable: road_condition (range 0-10; 0 = dry, 10 = wet)
road_condition = ctrl.Antecedent(np.arange(0, 11, 1), 'road_condition')

# Define output variable: brake (range 0-100%; 0 = no braking, 100 = full brake)
brake = ctrl.Consequent(np.arange(0, 101, 1), 'brake')

"""Define fuzzy membership sets."""
# Speed can be 'slow', 'medium', or 'fast'
# fuzz.trimf() creates a triangular membership function over the universe of values
speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 60])
speed['medium'] = fuzz.trimf(speed.universe, [40, 70, 100])
speed['fast'] = fuzz.trimf(speed.universe, [80, 120, 120])

# Distance can be 'close', 'medium', or 'far'
distance['close'] = fuzz.trimf(distance.universe, [0, 0, 40])
distance['medium'] = fuzz.trimf(distance.universe, [20, 50, 80])
distance['far'] = fuzz.trimf(distance.universe, [60, 100, 100])

# Road condition can be 'dry', 'damp', or 'wet'
road_condition['dry'] = fuzz.trimf(road_condition.universe, [0, 0, 4])
road_condition['damp'] = fuzz.trimf(road_condition.universe, [2, 5, 8])
road_condition['wet'] = fuzz.trimf(road_condition.universe, [6, 10, 10])

# Brake force can be 'light', 'medium', or 'hard'
brake['light'] = fuzz.trimf(brake.universe, [0, 0, 40])
brake['medium'] = fuzz.trimf(brake.universe, [30, 50, 70])
brake['hard'] = fuzz.trimf(brake.universe, [60, 100, 100])

"""Define fuzzy roles."""
# Rules combine fuzzy sets using logical AND
# The result defines how the system reacts to differenct conditions

# Rule 1: If speed is fast AND distance is close AND road is dry -> hard brake
rule1 = ctrl.Rule(speed['fast'] & distance['close'] & road_condition['dry'], brake['hard'])

# Rule 2: If speed is fast AND distance is medium AND road is dry -> medium brake
rule2 = ctrl.Rule(speed['fast'] & distance['medium'] & road_condition['dry'], brake['medium'])

# Rule 3: If speed is fast AND road is wet -> hard brake
rule3 = ctrl.Rule(speed['fast'] & road_condition['wet'], brake['hard'])

# Rule 4: If speed is medium AND distance is close AND road is wet -> hard brake
rule4 = ctrl.Rule(speed['medium'] & distance['close'] & road_condition['wet'], brake['hard'])

# Rule 5: If speed is medium AND distance is medium AND road is damp -> medium brake
rule5 = ctrl.Rule(speed['medium'] & distance['medium'] & road_condition['damp'], brake['medium'])

# Rule 6: If speed is medium AND distance is far AND road is dry -> light brake
rule6 = ctrl.Rule(speed['medium'] & distance['far'] & road_condition['dry'], brake['light'])

# Rule 7: If speed is slow AND distance is close AND road is wet -> brake medium
rule7 = ctrl.Rule(speed['slow'] & distance['close'] & road_condition['wet'], brake['medium'])

# Rule 8: If speed is slow AND distance is far -> light brake
rule8 = ctrl.Rule(speed['slow'] & distance['far'], brake['light'])

"""Build control system."""
# Create a control system from all the defined rules
braking_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8
])
# Create a simulation object for this control system
braking_sim = ctrl.ControlSystemSimulation(braking_ctrl)

"""Provide input values for simulation."""
# Example inputs
speed_value = 90       # km/h
distance_value = 30    # meters
road_value = 8         # wet (0–10)

# Assign the numeric inputs to the fuzzy system
braking_sim.input['speed'] = speed_value     # km/h
braking_sim.input['distance'] = distance_value     # meters
braking_sim.input['road_condition'] = road_value          # wet (scale 0–10)

# Performs all the fuzzification, rule evaluation, and defuzzification steps
braking_sim.compute()

"""Display the results."""
print(f"Speed: {speed_value} km/h")
print(f"Distance: {distance_value} m")
print(f"Road condition: {road_value} (wetness level)")
print(f"Recommended braking force: {braking_sim.output['brake']:.2f}%") # .2f to the second decimal place

"""Visualize membership and results."""
speed.view(sim=braking_sim)
distance.view(sim=braking_sim)
road_condition.view(sim=braking_sim)
brake.view(sim=braking_sim)