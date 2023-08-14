import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils import load_images_from_folders


class EigenFaces:
    def __init__(self, m=20, n_neighbors=4):
        self.x, y = load_images_from_folders()
        self.__faces = np.reshape(self.x, (self.x.shape[0], 56 * 46))

        self.__mean_face = np.mean(self.__faces, axis=0)

        self.__w = self.__get_weights(faces=self.__faces, m=m)

        # Classifier
        x = np.array([self.__get_projection(face) for face in self.__faces])
        self.__classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.__classifier.fit(x, y)

    def get_weights(self):
        return self.__w.copy()

    def __get_weights(self, faces, m):
        # get weights
        a = faces - self.__mean_face
        a = a.T
        r = np.dot(a.T, a)

        epsilon, values = np.linalg.eig(r)

        sorted_idx = np.argsort(epsilon)[::-1][:m]
        values = values[:, sorted_idx]

        return np.dot(a, values)

    def __get_projection(self, face):
        face_subspace = np.reshape(face, (56 * 46))
        projection = np.dot(self.__w.T, face_subspace - self.__mean_face)
        return np.real(projection)  # weighted sum slide 7

    def get_prediction(self, face):
        projection = self.__get_projection(face)
        prediction = self.__classifier.predict([projection])
        return prediction[0]

    def get_reconstruction(self, face):
        face_subspace = np.reshape(face, (56 * 46))
        projection = np.dot(self.__w.T, face_subspace - self.__mean_face)
        reconstruction_subspace = np.dot(self.__w, projection) + self.__mean_face
        reconstruction = np.reshape(reconstruction_subspace, (56, 46))

        return reconstruction

    def get_results(self):
        n_img = len(self.__faces)
        size = int(np.ceil(np.sqrt(n_img)))

        plt.figure()
        for i in range(n_img):
            plt.subplot(size, size, i + 1)
            plt.imshow(self.x[i], cmap='gray')
            plt.axis("off")
        plt.show()

        errors = []

        plt.figure()
        for i in range(n_img):
            reconstruction = self.get_reconstruction(self.x[i])
            errors.append(self.x[i] - reconstruction)
            plt.subplot(size, size, i + 1)
            plt.imshow(reconstruction, cmap='gray')
            plt.axis("off")
        plt.show()

        plt.figure()
        for i in range(n_img):
            plt.subplot(size, size, i + 1)
            plt.imshow(errors[i], cmap='gray')
            plt.axis("off")
        plt.show()
