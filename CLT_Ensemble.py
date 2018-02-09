import numpy as np
from scipy.stats import gaussian_kde
from time import strftime


class CLT_Ensemble:

    def __init__(self, algorithms: list, training_images: np.ndarray, training_labels: np.ndarray):
        self.algorithms = algorithms
        self.models = [alg.model for alg in algorithms]
        self.training_images = training_images
        self.training_labels = training_labels
        self.accuracy = None
        self.last_good_fisherfaces_data = (training_images, training_labels)


    def update_models(self, faces: np.ndarray, labels: np.ndarray, num_passes: int = 3, batch_size: int = 20):

        for _ in range(num_passes):
            print(strftime("[%Y.%m.%d %H:%M:%S]") + " Updating models... pass " + str(_))

            self.update_kernel_density(faces)

            agreements = []
            for face, label in zip(faces, labels):
                predictions = [model.predict(face) for model in self.models]
                predicted_labels = np.array(list(map(lambda x:x[0], predictions)))

                # if all algorithms predicted the same face, update training data and retrain models
                if np.all(predicted_labels[0] == predicted_labels):
                    #print(strftime("[%Y.%m.%d %H:%M:%S]") + " Retraining models...")
                    #self.training_images = np.concatenate((self.training_images, face.reshape(1, *face.shape)), axis=0)
                    #self.training_labels = np.concatenate((self.training_labels, np.array([label])), axis=0)
                    agreements.append((sum([self.algorithms[i].confidence(predictions[i][1]) for i in range(len(predictions))]), face.reshape(1, *face.shape), np.array([label])))

            agreements = sorted(agreements, key=lambda x:-x[0])
            for agreement in range(0, len(agreements), batch_size):
                #print(strftime("[%Y.%m.%d %H:%M:%S]") + " Processing agreement %d/%d" % (agreement, len(agreements)))
                new_training_images = np.asarray(list(map(lambda x:x[1].reshape((x[1].shape[1], x[1].shape[2])), agreements[agreement:agreement + batch_size])))
                new_training_labels = np.asarray(list(map(lambda x:x[2], agreements[agreement:agreement + batch_size]))).reshape((new_training_images.shape[0]))
                self.training_images = np.concatenate((self.training_images, new_training_images), axis=0)
                self.training_labels = np.concatenate((self.training_labels, new_training_labels), axis=0)
                self.training_images, indices = np.unique(self.training_images, return_index=True, axis=0)
                self.training_labels = self.training_labels[indices]
                for algorithm in range(len(self.algorithms)):
                    try:
                        self.algorithms[algorithm].train(self.training_images, self.training_labels, do_print=False)
                        if self.algorithms[algorithm].__class__.NAME == "Fisherfaces":
                            self.last_good_fisherfaces_data = (self.training_images, self.training_labels)
                    except:
                        self.algorithms[algorithm].train(*self.last_good_fisherfaces_data, do_print=False)


    def test(self, faces: np.ndarray, labels: np.ndarray) -> (float, list):
        num_correct = 0.
        correct_indices = []
        self.update_kernel_density(faces)
        for face, label in zip(faces, labels):

            predictions = [model.predict(face) for model in self.models]
            predicted_labels = np.array(list(map(lambda x:x[0], predictions)))

            distances = list(map(lambda x:x[1], predictions))

            confidences = [self.algorithms[i].confidence(distances[i]) for i in range(len(self.algorithms))]

            bins = dict.fromkeys(labels, 0)
            for i in range(len(predicted_labels)):
                bins[predicted_labels[i]] += confidences[i]
            guess = max(bins, key=bins.get)

            if guess == label:
                num_correct += 1
                correct_indices.append(1)
            else:
                correct_indices.append(0)
        self.accuracy = 100*num_correct/len(faces)
        return self.accuracy, correct_indices


    def update_kernel_density(self, faces: np.ndarray) -> None:
        # recalculate all distances
        for algorithm in range(len(self.algorithms)):
            self.algorithms[algorithm].distances = [self.algorithms[algorithm].model.predict(faces[face, :, :])[1] for face in
                                                    range(faces.shape[0])]

        # recalculate kernel density estimate
        for algorithm in self.algorithms:
            algorithm.pdf = gaussian_kde(algorithm.distances)