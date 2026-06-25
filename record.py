import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GestureRecognition"))

from labeling import data_labeling

GESTURES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
PER_PERSON = 6

print("Jede Person nimmt 6 Aufnahmen pro Geste auf (5x6=30 neue pro Geste).\n")
print(f"Gesten: {', '.join(GESTURES)}  ({len(GESTURES)} Stueck)\n")

for gesture in GESTURES:
    print(f"\n{'='*40}")
    print(f"  Geste: {gesture}  --  {PER_PERSON} Aufnahmen")
    print(f"{'='*40}")
    data_labeling(times=PER_PERSON, label=gesture)

print("\nFertig! Naechste Person kann starten.")
