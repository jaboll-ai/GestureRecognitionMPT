import sys
sys.path.append("GestureRecognition")

import numpy as np
from hmmclassifier import HMMClassifier

classifier = HMMClassifier(n_states=3, n_iter=5)

# synthetische Sequenzen für eine Klasse
sequences = [np.random.randn(30, 2) for _ in range(5)]

A, pi, means, covs = classifier.baum_welch(sequences)

print("A shape:", A.shape)        # erwartet: (3, 3)
print("pi shape:", pi.shape)      # erwartet: (3,)
print("means shape:", means.shape)  # erwartet: (3, 2)
print("covs shape:", covs.shape)    # erwartet: (3, 2, 2)
print("pi summiert zu:", pi.sum())  # erwartet: ~1.0

# Funktioniert, shapes korrekt und pi summiert zu 1.0