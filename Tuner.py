from Algorithm import Algorithm
from Dataset import Dataset

from scipy.optimize import differential_evolution
from time import strftime

class Tuner:


    def __init__(self, algorithm_class: "class", dataset: Dataset, algorithm_param_names: list, **DEkwargs):
        """
        Tuner class to facilitate parameter tuning via differential evolution

        :param algorithm_class: handle to the class to train, must be child of Algorithm class
        :param dataset: dataset to train on, must be child of Dataset class
        :param algorithm_param_names: list of parameter kwargs for algorithm_class, such as "num_components"
        :param DEkwargs: keyword arguments for scipy differential evolution function
        """
        self.cls = algorithm_class
        self.dataset = dataset
        self.algorithm_param_names = algorithm_param_names
        self.DEparams = DEkwargs


    def tune(self, bounds: list) -> list:
        """
        Run scipy differential evolution to choose parameters that maximize accuracy

        :param bounds: list of tuples containing upper and lower bounds for each parameter
        :return:
        """

        print(strftime("[%Y.%m.%d %H:%M:%S]") + " Begin tuning algorithm " + self.cls.__name__ + "...")

        #
        result = differential_evolution(self.objective_function, bounds, [self.dataset], **self.DEparams)
        params, accuracy = list(map(int, result.x)), 100-result.fun
        print(strftime("[%Y.%m.%d %H:%M:%S] ") + self.cls.__name__ + " optimal parameters:\n\t" +
              "\n\t".join([self.algorithm_param_names[i] + ": " + str(params[i]) for i in range(len(result.x))]) +
              "\n\tAccuracy: %.2f%%" % (100 - result.fun))
        return params


    def objective_function(self, values: list, *args):
        """
        Objective function used in optimization.

        :param values:
        :param args:
        :return:
        """
        training, testing = self.dataset.get_training_faces()
        training_faces, training_labels = training
        testing_faces, testing_labels = testing
        kwargs = {}
        for i in range(len(values)):
            # warning: assumes parameter values are integers
            kwargs[self.algorithm_param_names[i]] = int(values[i])
        algo = self.cls(**kwargs)
        algo.train(training_faces, training_labels, do_print=False)
        algo.test(testing_faces, testing_labels)
        return 100 - algo.accuracy
