import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks import python as mp_python
import numpy as np
import cv2


class HandDetector: 

    inputSignals = ["frame"]
    outputSchema = {
        "detector": {
            "detected": bool,
            "landmarks": None
        }
    }

    def __init__(self, config=None):
        self.config = config or {}
        self.landmarker = None
        self.latest_result = None

    def start(self):
        import urllib.request, os
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Lade Modell herunter...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path
            )
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=1
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def stop(self):
        if self.landmarker:
            self.landmarker.close()

    def step(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand],
                dtype=np.float32
            )
            return {"detected": True, "landmarks": landmarks}
        else:
            return {"detected": False, "landmarks": np.zeros((21, 3), dtype=np.float32)}