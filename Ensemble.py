import numpy as np
from scipy.stats import mode


class Ensemble:

    def __init__(self, algorithms: list):
        self.algorithms = algorithms
        self.models = [alg.model for alg in algorithms]
        self.accuracy = None


    def test(self, faces: np.ndarray, labels: np.ndarray):
        num_correct = 0.
        correct_indices = []
        for face, label in zip(faces, labels):
            predictions = [model.predict(face)[0] for model in self.models]
            guess = mode(predictions, axis=None)[0]
            if guess == label:
                num_correct += 1
                correct_indices.append(1)
            else:
                correct_indices.append(0)
        self.accuracy = 100*num_correct/len(faces)
        return self.accuracy, correct_indices