from SignalHub import Module, GALY, get_nested_key
from collections import deque


class TrailMarker(Module):

    def __init__(self, outputSignal="trailmarker"):
        self.outputSignal = outputSignal

        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={
                "type": "object",
                "properties": {
                    outputSignal: {},
                    "galy": {}
                },
                "additionalProperties": True
            },
            name="trailmarker",
        )

    def start(self, data):

        config = data.get("config", {})

        # ✅ FIX: richtige Reihenfolge (key, data, default)
        self.finger_index = get_nested_key(
            "trailmarker.fingerIndex", config, 8
        )

        self.max_points = get_nested_key(
            "trailmarker.maxPoints", config, 50
        )

        self.max_lost_frames = get_nested_key(
            "trailmarker.maxLostFrames", config, 10
        )

        self.trail = deque(maxlen=self.max_points)
        self.lost_frames = 0

        return {}

    def _draw_trail(self, galy):
        # gleiche Pixelauflösung wie handdetector.draw_hand_landmarks
        width, height = 1280, 720
        galy.layer("trail")
        points = list(self.trail)
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            galy.line(pt1, pt2, (0, 255, 255), 2)

    def step(self, data):

        detector = data.get("detector", None)

        # Jedes zeichnende Modul erzeugt sein eigenes GALY-Objekt (wie
        # HandDetector) - die Engine benennt "galy"-Signale beim Mergen
        # automatisch eindeutig um, ein hereingereichtes "galy" aus data
        # existiert hier nie (TrailMarker abonniert dieses Signal nicht).
        galy = GALY()

        # --------------------------------------------------
        # 1. Hands extrahieren (robust für SignalHub/MediaPipe)
        # --------------------------------------------------
        hands = []

        if detector is None:
            hands = []
        else:
            # MediaPipe HandLandmarkerResult
            if hasattr(detector, "hand_landmarks"):
                hands = detector.hand_landmarks or []

        # --------------------------------------------------
        # 2. Keine Hand erkannt
        # --------------------------------------------------
        if len(hands) == 0:
            self.lost_frames += 1

            if self.lost_frames > self.max_lost_frames:
                self.trail.clear()

            self._draw_trail(galy)

            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        # --------------------------------------------------
        # 3. Hand erkannt
        # --------------------------------------------------
        self.lost_frames = 0

        hand = hands[0]

        # hand kann dict, Liste von Landmarken (MediaPipe) ODER Objekt sein → absichern
        if isinstance(hand, dict):
            landmarks = hand.get("landmarks", [])
        elif isinstance(hand, (list, tuple)):
            landmarks = hand
        else:
            landmarks = getattr(hand, "landmarks", [])

        if not landmarks or len(landmarks) <= self.finger_index:
            self._draw_trail(galy)
            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        landmark = landmarks[self.finger_index]

        # landmark kann dict ODER object sein
        if isinstance(landmark, dict):
            x = landmark.get("x")
            y = landmark.get("y")
        else:
            x = getattr(landmark, "x", None)
            y = getattr(landmark, "y", None)

        if x is None or y is None:
            self._draw_trail(galy)
            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        # --------------------------------------------------
        # 4. Trail updaten
        # --------------------------------------------------
        point = (x, y)
        self.trail.append(point)

        self._draw_trail(galy)

        return {
            self.outputSignal: {
                "trail": list(self.trail)
            },
            "galy": galy
        }

    def stop(self, data):
        self.trail.clear()
