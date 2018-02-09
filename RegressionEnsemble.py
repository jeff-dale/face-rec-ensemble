import numpy as np


class RegressionEnsemble:

    def __init__(self, models: list):
        self.models = models
        self.accuracy = None

    def test(self, faces: np.ndarray, labels: np.ndarray, coefs: np.ndarray):
        num_correct = 0.
        correct_indices = []
        for face, label in zip(faces, labels):
            bins = dict.fromkeys(labels, 0)
            for i in range(len(self.models)):
                bins[self.models[i].predict(face)[0]] += coefs[i]
            guess = max(bins, key=bins.get)
            if guess == label:
                num_correct += 1
                correct_indices.append(1)
            else:
                correct_indices.append(0)
        self.accuracy = 100*num_correct/len(faces)
        return self.accuracy, correct_indices