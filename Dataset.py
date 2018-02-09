import cv2
from glob import glob
import numpy as np
from os import walk
from random import randint


class Dataset:

    NAME = "Generic Dataset"

    def __init__(self, base_path: str):
        self.base_path = base_path

    def get_training_faces(self):
        raise NotImplementedError

    @staticmethod
    def read_facescrub_subjects(base_path: str) -> list:
        subjects = next(walk(base_path))[1]
        return subjects


    @staticmethod
    def random_facescrub_faces(base_path: str, faces_per_subject: int, dsize: (int, int), used: list = None) -> list:
        subjects = Dataset.read_facescrub_subjects(base_path)
        used = used if used is not None else []
        for subject in range(len(subjects)):
            face_images = glob(base_path + subjects[subject] + "/face/*.jpg")
            for _ in range(faces_per_subject):
                face_index = randint(0, len(face_images) - 1)
                while (subject, face_index) in used:
                    face_index = randint(0, len(face_images) - 1)
                used.append((subject, face_index))
                try:
                    face = cv2.imread(face_images[face_index], cv2.IMREAD_GRAYSCALE)
                    sized_face = cv2.resize(src=face, dsize=dsize)
                    yield sized_face, subject, used
                except:
                    print("OpenCV threw error on " + subjects[subject] + "/" + str(face_index))


    @staticmethod
    def split_dataset(faces: np.ndarray, labels: np.ndarray, ratio: float) \
            -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):

        test_size = int(ratio * len(faces))
        training_data, testing_data = faces[test_size:], faces[:test_size]
        training_labels, testing_labels = labels[test_size:], labels[:test_size]
        return (training_data, training_labels), (testing_data, testing_labels)