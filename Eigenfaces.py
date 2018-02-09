from Algorithm import Algorithm

import cv2
import numpy as np


class Eigenfaces(Algorithm):

    NAME = "Eigenfaces"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if "num_components" in kwargs:
            model = cv2.face.EigenFaceRecognizer_create(**kwargs)
        else:
            model = cv2.face.EigenFaceRecognizer_create()
        super().__init__(model)


    def train(self, faces: np.ndarray, labels: np.ndarray, **kwargs) -> None:
        super().train(faces, labels, Eigenfaces.NAME, **kwargs)
