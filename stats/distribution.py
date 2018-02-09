from ATT_Faces import ATT_Faces
from Yale_Faces import Yale_Faces

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kstest, probplot

ratio = 0.75

dataset = ATT_Faces("../datasets/att_faces/")
faces, labels = dataset.get_all_faces_and_labels()

flat_faces = []
for face in faces:
    flat_faces.append(face.flatten())

flat_faces = np.array(flat_faces).T[np.random.permutation(len(flat_faces[0])), :]

train_faces = flat_faces[:, :int(ratio*flat_faces.shape[1])]
test_faces = flat_faces[:, int(ratio*flat_faces.shape[1]):]


x = 0
