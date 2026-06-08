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
        self.hands = None

    def start(self):
        pass

    def stop(self):
        if self.hands:
            self.hands.close()

    def step(self, frame):
        pass