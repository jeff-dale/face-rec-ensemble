# training_faces = np.empty((training_faces_per_subject * len(subjects), *face_size), dtype=np.uint8)
# training_labels = []
# current_face_index = 0
# used = None
# for face, label, used in Dataset.random_facescrub_faces(base_face_path, training_faces_per_subject, dsize=face_size):
#    print("Getting training face %d/%d" % (current_face_index + 1, training_faces_per_subject*len(subjects)))
#    training_faces[current_face_index, :, :] = face
#    training_labels.append(label)
#    current_face_index += 1
# training_faces = training_faces[np.random.permutation(training_faces.shape[0]), :, :]
# training_labels = np.asarray(training_labels)

# testing_faces = np.empty((testing_faces_per_subject * len(subjects), *face_size), dtype=np.uint8)
# testing_labels = []
# current_face_index = 0
# for face, label, used in Dataset.random_facescrub_faces(base_face_path, testing_faces_per_subject, dsize=face_size, used=used):
#    print("Getting testing face %d/%d" % (current_face_index + 1, testing_faces_per_subject*len(subjects)))
#    testing_faces[current_face_index, :, :] = face
#    testing_labels.append(label)
#    current_face_index += 1
#    testing_faces = testing_faces[np.random.permutation(testing_faces_per_subject * len(subjects)), :, :]
# testing_labels = np.asarray(testing_labels)