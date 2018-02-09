import numpy as np
from scipy.integrate import quadrature
from time import strftime, time


class Algorithm:

    def __init__(self, model=None, accuracy=None):
        self.model = model
        self.accuracy = accuracy
        self.distances = None
        self.correct_indices = None
        self.median_distance = None
        self.std = None
        self.correct_num = None
        self.pdf = None


    def confidence(self, guess_distance: float) -> float:
        return quadrature(lambda t: 1.0/((1-t)*(1-t))*self.pdf.evaluate(guess_distance + t/(1-t)), 0, 1, maxiter=50)[0]


    def train(self, faces: np.ndarray, labels: np.ndarray, algorithm_name: str, do_print: bool = False):
        if do_print: print("[%s] Training %s model..." % (strftime("%Y.%m.%d %H:%M:%S"), algorithm_name))
        start_time = time()
        self.model.train(faces, labels)
        if do_print: print("[%s] Training on model %s finished in %.3f seconds" % (strftime("%Y.%m.%d %H:%M:%S"), algorithm_name, time() - start_time))
        return self


    def test(self, testing_faces: np.ndarray, testing_labels: np.ndarray) -> None:
        num_correct = 0.
        correct_indices = []
        distances = []
        for face, label in zip(testing_faces, testing_labels):
            predicted_label = self.model.predict(face)
            if predicted_label[0] == label:
                num_correct += 1
                correct_indices.append(1)
            else:
                correct_indices.append(0)
            distances.append(predicted_label[1])
        distances = np.array(distances)

        #import matplotlib.pyplot as plt; plt.scatter(distances, np.zeros_like(distances) + correct_indices, c=np.asarray(["r" if i not in np.where(correct_indices)[0] else "g" for i in range(len(distances))])); plt.show()
        self.correct_num = num_correct
        self.median_distance = np.median(distances)
        self.std = np.std(distances)
        self.accuracy = 100.0*num_correct/len(testing_faces)
        self.distances = distances
        self.correct_indices = correct_indices

        #import matplotlib.pylab as pylab; from scipy.stats import probplot; probplot(self.distances, dist="norm", plot=pylab)
        #pylab.show()
