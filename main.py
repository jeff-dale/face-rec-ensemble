from CLT_Ensemble import CLT_Ensemble
from Eigenfaces import Eigenfaces
from Ensemble import Ensemble
from Fisherfaces import Fisherfaces
from LocalBinaryPatternHistograms import LocalBinaryPatternHistograms
from RegressionEnsemble import RegressionEnsemble
from Tuner import Tuner

# noinspection PyUnresolvedReferences
from ATT_Faces import ATT_Faces

# noinspection PyUnresolvedReferences
from Yale_Faces import Yale_Faces

import numpy as np
import sys
from threading import Thread
from time import strftime


class Tee(object):
    """
    For writing to file and stdout simultaneously
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()


if __name__ == "__main__":

    sys.stdout = Tee("logs/execution log - " + strftime("%Y.%m.%d %H.%M.%S.txt"), "w")
    np.warnings.filterwarnings('ignore')

    dataset = ATT_Faces("datasets/att_faces/")
    #dataset = Yale_Faces("datasets/ext_yale_b/CroppedYale/")
    #Yale_Faces.create_index_csv()

    # define some constants
    SHOULD_TUNE = False
    QUIT_AFTER_TUNING = True
    num_passes = 5

    eigenfaces_params_names = ["num_components"]
    fisherfaces_params_names = ["num_components"]
    lbph_params_names = ["radius", "neighbors", "grid_x", "grid_y"]

    # read images
    print("Reading images from: " + dataset.NAME)

    if SHOULD_TUNE:
        print("================================================== Parameter Tuning ==================================================")
        # TODO - tune on different dataset
        training, testing = dataset.get_training_faces()
        training_faces, training_labels = training
        testing_faces, testing_labels = testing
        eigenfaces_params = Tuner(Eigenfaces, dataset, eigenfaces_params_names, maxiter=10) \
            .tune([(1, 100)])
        fisherfaces_params = Tuner(Fisherfaces, dataset, fisherfaces_params_names, maxiter=10) \
            .tune([(1, 100)])
        lbph_params = Tuner(LocalBinaryPatternHistograms, dataset, lbph_params_names, maxiter=5) \
            .tune([(1, 5), (1, 12), (1, 12), (1, 12)])

        print()
        if QUIT_AFTER_TUNING: exit(0)

    else:
        # ================================
        # AT&T Params
        # eigenfaces_params = [59]
        # fisherfaces_params = [46]
        # lbph_params = [4, 9, 2, 5]
        # ================================
        # Yale Params
        eigenfaces_params = [82]
        fisherfaces_params = [31]
        lbph_params = [4, 11, 11, 11]

        print(strftime("[%Y.%m.%d %H:%M:%S] ") + "Using user-supplied parameters...")
        for cls, algorithm_param_names, algorithm_param_values in zip(
                [Eigenfaces, Fisherfaces, LocalBinaryPatternHistograms],
                [eigenfaces_params_names, fisherfaces_params_names, lbph_params_names],
                [eigenfaces_params, fisherfaces_params, lbph_params]):

            print(strftime("[%Y.%m.%d %H:%M:%S] ") + cls.__name__ + " parameters:\n\t" +
                  "\n\t".join([algorithm_param_names[i] + ": " + str(algorithm_param_values[i]) for i in range(len(algorithm_param_values))]) +
                  "\n\tAccuracy: Unknown")

    training, testing = dataset.get_training_faces()
    training_faces, training_labels = training
    testing_faces, testing_labels = testing

    # prepare models
    eigenfaces = Eigenfaces(num_components=eigenfaces_params[0])
    fisherfaces = Fisherfaces(num_components=fisherfaces_params[0])
    lbph = LocalBinaryPatternHistograms(**{param: value for param, value in zip(["radius", "neighbors", "grid_x", "grid_y"], lbph_params)})

    # train models, can't use multiprocessing because OpenCV objects won't pickle. Using threading helps boost cpu usage slightly (I think)
    print(strftime("[%Y.%m.%d %H:%M:%S] ") + "Training initial models")
    eigenfaces_proc = Thread(target=eigenfaces.train, args=(training_faces, training_labels)); eigenfaces_proc.start()
    fisherfaces_proc = Thread(target=fisherfaces.train, args=(training_faces, training_labels)); fisherfaces_proc.start()
    lbph_proc = Thread(target=lbph.train, args=(training_faces, training_labels)); lbph_proc.start()

    eigenfaces_proc.join()
    fisherfaces_proc.join()
    lbph_proc.join()

    print()

    # predict using training data
    eigenfaces.test(testing_faces, testing_labels)
    print("Eigenfaces accuracy: \t%.2f%% (%d/%d)\t\t%s" % (eigenfaces.accuracy, sum(eigenfaces.correct_indices), len(testing_labels), ",".join(list(map(str, eigenfaces.correct_indices)))))

    fisherfaces.test(testing_faces, testing_labels)
    print("Fisherfaces accuracy: \t%.2f%% (%d/%d)\t\t%s" % (fisherfaces.accuracy, sum(fisherfaces.correct_indices), len(testing_labels), ",".join(list(map(str, fisherfaces.correct_indices)))))

    lbph.test(testing_faces, testing_labels)
    print("LBPH accuracy: \t\t\t%.2f%% (%d/%d)\t\t%s" % (lbph.accuracy, sum(lbph.correct_indices), len(testing_labels), ",".join(list(map(str, lbph.correct_indices)))))


    # faces that were correctly guessed by at least one algorithm
    guessed_faces = [(1 if i + j + k > 0 else 0) for i, j, k in zip(eigenfaces.correct_indices, fisherfaces.correct_indices, lbph.correct_indices)]
    ensemble = Ensemble([eigenfaces, fisherfaces, lbph])
    regression_ensemble = RegressionEnsemble([eigenfaces.model, fisherfaces.model, lbph.model])
    _, ensemble_indices = ensemble.test(testing_faces, testing_labels)

    print("Ensemble accuracy: \t\t%.2f%% (%d/%d)\t\t%s" % (ensemble.accuracy, sum(ensemble_indices), len(testing_labels), ",".join(list(map(str, ensemble_indices)))))
    print("Theoretical accuracy: \t%.2f%% (%d/%d)\t\t%s" % (100 * sum(guessed_faces) / len(guessed_faces), sum(guessed_faces), len(testing_labels),",".join(list(map(str, guessed_faces)))))
    print()

    print("====================================================== Ensemble =======================================================")

    print(strftime("[%Y.%m.%d %H:%M:%S] ") + "Running ensemble method...")
    for passes in range(num_passes + 1):
        clt_ensemble = CLT_Ensemble([eigenfaces, fisherfaces, lbph], training_faces, training_labels)
        clt_ensemble.update_models(testing_faces, testing_labels, num_passes=passes, batch_size=10000)
        _, clt_ensemble_indices = clt_ensemble.test(testing_faces, testing_labels)
        print("CLT %d passes accuracy: \t%.2f%% (%d/%d)\t\t%s" % (passes, clt_ensemble.accuracy, sum(clt_ensemble_indices), len(testing_labels), ",".join(list(map(str, clt_ensemble_indices)))))

    print()
