from Algorithm import Algorithm

import cv2
import numpy as np


class LocalBinaryPatternHistograms(Algorithm):

    NAME = "local binary pattern histograms"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        model = cv2.face.LBPHFaceRecognizer_create(**kwargs)
        super().__init__(model)


    def train(self, faces: np.ndarray, labels: np.ndarray, **kwargs) -> None:
        super().train(faces, labels, LocalBinaryPatternHistograms.NAME, **kwargs)