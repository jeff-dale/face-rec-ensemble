from Dataset import Dataset

import cv2
import numpy as np
from random import shuffle

class ATT_Faces(Dataset):

    NAME = "AT&T Faces"

    def __init__(self, base_path: str):
        self.base_path = base_path
        super().__init__(base_path)


    def get_training_faces(self, faces_per_subject: int = 1) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):

        ## read index file
        #with open(self.base_path + "faces.csv", "r") as f:
        #    lines = list(map(lambda x: x.split(";"), f.readlines()))
#
        ## get all faces and labels
        #faces = np.asarray(list(map(lambda x: cv2.imread(self.base_path + x[0], cv2.IMREAD_GRAYSCALE), lines)))
        #labels = np.array(list((map(lambda x: int(x[1]), lines))))

        faces, labels = self.get_all_faces_and_labels()

        # get the unique labels of faces
        unique_labels = np.unique(labels)

        # build training set with one random face for each subject
        training_indices = [np.random.choice(np.where(labels == i)[0]) for i in unique_labels]
        training_permutation = np.random.permutation(len(unique_labels))
        training_faces, training_labels = faces[training_indices][training_permutation], labels[training_indices][training_permutation]

        # use rest of dataset as testing
        testing_indices = np.delete(range(len(labels)), training_indices)
        testing_permutation = np.random.permutation(len(testing_indices))
        testing_faces, testing_labels = faces[testing_indices][testing_permutation], labels[testing_indices][testing_permutation]

        return (training_faces, training_labels), (testing_faces, testing_labels)


    def get_all_faces_and_labels(self) -> (np.ndarray, np.ndarray):
        # read index file
        with open(self.base_path + "faces.csv", "r") as f:
            lines = list(map(lambda x: x.split(";"), f.readlines()))

        # get all faces and labels
        faces = np.asarray(list(map(lambda x: cv2.imread(self.base_path + x[0], cv2.IMREAD_GRAYSCALE), lines)))
        labels = np.array(list((map(lambda x: int(x[1]), lines))))

        return faces, labels


    def faces_per_subject(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):

        # read index file
        with open(self.base_path + "faces.csv", "r") as f:
            lines = list(map(lambda x: x.split(";"), f.readlines()))

        # get all faces and labels
        faces = np.asarray(list(map(lambda x: cv2.imread(self.base_path + x[0], cv2.IMREAD_GRAYSCALE), lines)))
        labels = np.array(list((map(lambda x: int(x[1]), lines))))

        # get the unique labels of faces
        unique_labels = np.unique(labels)

        # build training set with one random face for each subject
        training_indices = [np.random.choice(np.where(labels == i)[0]) for i in unique_labels]
        training_permutation = np.random.permutation(len(unique_labels))
        training_faces, training_labels = faces[training_indices][training_permutation], labels[training_indices][training_permutation]

        # use rest of dataset as testing
        testing_indices = np.delete(range(len(labels)), training_indices)
        testing_permutation = np.random.permutation(len(testing_indices))
        testing_faces, testing_labels = faces[testing_indices][testing_permutation], labels[testing_indices][testing_permutation]

        return (training_faces, training_labels), (testing_faces, testing_labels)
