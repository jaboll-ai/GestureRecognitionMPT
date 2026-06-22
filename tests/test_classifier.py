import sys
sys.path.append("GestureRecognition")

import numpy as np
from hmmclassifier import HMMClassifier

classifier = HMMClassifier(n_states=3, n_iter=5)

# synthetische Daten für 2 Klassen
sequences = [np.random.randn(30, 2) for _ in range(6)]
labels = ["A", "A", "A", "B", "B", "B"]

# Training
classifier.fit(sequences, labels)

# Vorhersage
test_seq = np.random.randn(30, 2)
scores = classifier.decision_function(test_seq)
prediction = classifier.predict(test_seq)

print("Scores:", scores)        # z.B. {"A": -120.5, "B": -145.3}
print("Vorhersage:", prediction) # "A" oder "B"

# Funktioniert, Scores und Vorhersage korrekt