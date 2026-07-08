import pickle
from datetime import datetime

from SignalHub import GALY, Module, bgr, get_nested_key

from GestureRecognition.augmentation import augment_recording
from GestureRecognition.labeling import (
    AUGMENTED_DATA_DIR,
    N_AUGMENTATIONS_PER_RECORDING,
    RAW_DATA_DIR,
    _next_sequence_number,
)


class GestureRecorder(Module):
    """
    Live-Pendant zu labeling.data_labeling(): speichert im Trigger-Modus
    (Taste 'a' startet/stoppt die Aufnahme) jede aufgenommene Trajektorie
    als neue Trainingsaufnahme für ``label`` und augmentiert sie danach
    automatisch, statt eine Vorhersage zu treffen.
    """

    def __init__(self, label, kuerzel=None, outputSignal="recorder"):
        self.label = label
        self.kuerzel = kuerzel
        self.outputSignal = outputSignal

        super().__init__(
            inputSignals=["config", "trigger", "preprocessor"],
            outputSchema={
                "type": "object",
                "properties": {outputSignal: {}, "galy": {}},
                "additionalProperties": True,
            },
            name="gesturerecorder",
        )

    def start(self, data):
        self.saved_count = 0
        self.label_dir = RAW_DATA_DIR / self.label
        self.label_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\nAufnahme-Modus für Label '{self.label}': "
            f"'a' = starten, nochmal 'a' = speichern & augmentieren\n"
        )
        return {}

    def _save(self, trajectory):
        date_str = datetime.now().strftime("%d-%m-%Y")
        seq = _next_sequence_number(self.label_dir)
        name_parts = [self.label] + ([self.kuerzel] if self.kuerzel else []) + [date_str, f"{seq:02d}"]
        path = self.label_dir / f"{'_'.join(name_parts)}.pkl"

        recording = {"preprocessor": [{"preprocessor": trajectory}]}
        with open(path, "wb") as f:
            pickle.dump(recording, f)

        self.saved_count += 1
        print(f"Gespeichert: {path}")

        augment_recording(
            path,
            output_dir=str(AUGMENTED_DATA_DIR),
            n_per_recording=N_AUGMENTATIONS_PER_RECORDING,
        )

    def step(self, data):
        trigger = data.get("trigger") or {}
        recording = trigger.get("recording", False)
        predict_now = trigger.get("predict_now", False)

        if predict_now:
            trajectory = get_nested_key("preprocessor", data)
            if trajectory is not None and len(trajectory) > 0:
                self._save(trajectory)
            else:
                print("Keine verwertbare Trajektorie - verworfen.")

        width  = get_nested_key("config.webcam.width",  data, default=1280)
        height = get_nested_key("config.webcam.height", data, default=720)

        if recording:
            text, color = "Aufnahme laeuft... ('a' zum Speichern)", bgr("#0080FF")
        else:
            text, color = f"Label '{self.label}': {self.saved_count} gespeichert - 'a' zum Starten", bgr("#AAAAAA")

        galy = GALY()
        galy.layer("recorder")
        galy.putText(
            text,
            (int(width * 0.05), int(height * 0.1)),
            fontScale=1.0,
            color=color,
        )

        return {self.outputSignal: {"saved": self.saved_count}, "galy": galy}

    def stop(self, data):
        pass
