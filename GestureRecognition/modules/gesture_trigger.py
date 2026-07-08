import threading

import keyboard
from SignalHub import Module


class GestureTrigger(Module):
    """
    Steuert Start/Stop einer Gesten-Aufnahme über EINE Taste (Toggle):
    erster Druck startet die Aufnahme, zweiter Druck stoppt sie und löst
    genau ein Ereignis (Prediction oder Speichern) für die gesammelte
    Trajektorie aus.

    Nutzt die ``keyboard``-Bibliothek für einen GLOBALEN Hotkey-Listener,
    der unabhängig davon funktioniert, welches Fenster gerade fokussiert
    ist (also auch während das Kamera-Fenster im Fokus ist).
    """

    def __init__(self, toggle_key="a", outputSignal="trigger"):
        self.outputSignal = outputSignal
        self.toggle_key = toggle_key.lower()

        super().__init__(
            inputSignals=[],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="gesturetrigger",
        )

    def start(self, data):
        self._recording = False
        self._predict_now = False
        self._lock = threading.Lock()

        keyboard.on_press_key(self.toggle_key, self._on_toggle_key)

        print(
            f"\nSteuerung (funktioniert überall, auch mit Fokus auf dem Kamera-Fenster):\n"
            f"  '{self.toggle_key}' drücken = Aufnahme starten\n"
            f"  '{self.toggle_key}' nochmal drücken = Aufnahme stoppen & auslösen\n"
        )

        return {}

    def _on_toggle_key(self, event):
        with self._lock:
            if not self._recording:
                self._recording = True
                print("Aufnahme gestartet - Geste ausführen ...")
            else:
                self._predict_now = True

    def step(self, data):
        with self._lock:
            recording = self._recording
            predict_now = self._predict_now

            # predict_now ist ein einmaliger Impuls für genau einen Frame,
            # danach zurück in den Ruhezustand
            if predict_now:
                self._predict_now = False
                self._recording = False

        return {
            self.outputSignal: {
                "recording": recording,
                "predict_now": predict_now,
            }
        }

    def stop(self, data):
        keyboard.unhook_all()
