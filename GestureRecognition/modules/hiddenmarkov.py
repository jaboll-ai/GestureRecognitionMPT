import numpy as np
from SignalHub import GALY, bgr, get_nested_key, Module
from GestureRecognition.hmmclassifier import HMMClassifier


class HMMModule(Module):
    """
    Modul zur Klassifikation von Gesten mittels Hidden Markov Models.

    Empfängt eine vorverarbeitete Fingertrajektorie vom Preprocessor,
    klassifiziert sie mit einem trainierten HMMClassifier und
    visualisiert das Ergebnis.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm_model.pkl", **kwargs):
        self.outputSignal = outputSignal
        self.model_path   = model_path

        super().__init__(
            name         = "hiddenmarkov",
            inputSignals = ["config", "preprocessor", "trigger"],
            outputSchema = {"type": "object", "properties": {outputSignal: {}}},
        )

    def start(self, data: dict) -> dict:
        model_path  = get_nested_key('config.hmm_model_path', data, default=self.model_path)
        self.model = HMMClassifier()
        self.model.load_model(model_path)
        self.last_result = None 
        return {}

    def step(self, data: dict) -> dict:
        # Optionales Trigger-Signal (GestureTrigger): nur vorhanden, wenn
        # das Programm mit --trigger gestartet wurde. Ohne Trigger-Modul
        # wird wie bisher jeden Frame neu vorhergesagt.
        trigger = data.get("trigger")
        gated = trigger is not None
        recording = trigger.get("recording", False) if gated else False
        should_predict = trigger.get("predict_now", False) if gated else True

        trajectory = get_nested_key('preprocessor', data)

        if should_predict and trajectory is not None and len(trajectory) > 0:

            seq        = np.array(trajectory, dtype=np.float32)
            best_label = self.model.predict([seq])[0]
            scores_arr = self.model.decision_function([seq])[0]
            best_score = float(np.max(scores_arr))

            self.last_result = {
                "label":  best_label,
                "score":  best_score,
                "scores": dict(zip(self.model.classes_, scores_arr)),
            }

        # Nichts zu zeigen
        if self.last_result is None and not gated:
            return {}

        # Letztes Ergebnis weiter anzeigen
        width  = get_nested_key('config.webcam.width',  data, default=1280)
        height = get_nested_key('config.webcam.height', data, default=720)

        galy = GALY()
        galy.layer("hmm")

        if self.last_result is not None:
            text, color = f"{self.last_result['label']}  {self.last_result['score']:.2f}", bgr("#AD0303")
        elif recording:
            text, color = "Aufnahme laeuft... ('a' zum Erkennen)", bgr("#0080FF")
        else:
            text, color = "Bereit - 'a' druecken zum Starten", bgr("#AAAAAA")

        galy.putText(
            text,
            (int(width * 0.05), int(height * 0.1)),
            fontScale = 1.3,
            color     = color,
        )

        # for i, (label, score) in enumerate(self.last_result["scores"].items()):
        #     galy.putText(
        #         f"{label}: {score:.2f}",
        #         (int(width * 0.05), int(height * 0.1) + 40 + i * 30),
        #         fontScale = 0.8,
        #         color     = bgr("#FFFFFF"),
        #     )

        return {self.outputSignal: self.last_result, "galy": galy}

    
    def stop(self, data: dict) -> None:
        pass