import sys
import numpy as np
sys.path.append("GestureRecognition")
from hmmclassifier import HMMClassifier

classifier = HMMClassifier()

seq = np.random.randn(30, 2)
N = 5

A = np.ones((N, N)) / N
pi = np.ones(N) / N
means = np.random.randn(N, 2)
covs = np.array([np.eye(2)] * N)

print(classifier.forward(seq, A, pi, means, covs))

# Funktioniert, kam bei mir jz -103.74089775357231 raus 