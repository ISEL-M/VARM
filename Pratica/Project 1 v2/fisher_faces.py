import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from eigen_faces import EigenFaces
from utils import load_images_from_folders


class FisherFaces:
    def __init__(self, m=20, n_neighbors=4):
        self.__x, self.__y = load_images_from_folders()
        self.__faces = np.reshape(self.__x, (self.__x.shape[0], 56 * 46))

        self.__mean_face = np.mean(self.__faces, axis=0)

        self.__unique_ys = np.unique(self.__y)
        self.__mean_faces = [np.mean(self.__faces[uni_y == self.__y], axis=0) for uni_y in self.__unique_ys]
        self.__n_i = [np.sum(uni_y == self.__y, axis=0) for uni_y in self.__unique_ys]

        self.__eigen_faces = EigenFaces(m)
        self.__w_pca = self.__eigen_faces.get_weights()

        self.__w = self.__get_weights(m=m)

        # Classifier
        x = np.array([self.__get_projection(face) for face in self.__faces])
        self.__classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.__classifier.fit(x, self.__y)

    def __get_weights(self, m):
        expected_s_t = np.ndarray([])
        aux = self.__faces - self.__mean_face
        for idx in range(len(aux)):
            aux2 = np.resize(aux[idx], (56 * 46, 1))
            expected_s_t = expected_s_t + np.dot(aux2, aux2.T)

        s_w = np.ndarray([])
        aux2 = self.__unique_ys.tolist()
        for idx in range(len(self.__faces)):
            aux3 = aux2.index(self.__y[idx])
            aux = self.__faces[idx] - self.__mean_faces[aux3]
            aux = np.reshape(aux, (56 * 46, 1))
            s_w = s_w + np.dot(aux, aux.T)

        s_b = np.ndarray([])
        aux = self.__mean_faces - self.__mean_face
        for idx in range(len(self.__unique_ys)):
            aux2 = np.resize(aux[idx], (56 * 46, 1))
            s_b = s_b + np.dot(aux2, aux2.T) * self.__n_i[idx]

        # s_t = s_w + s_b

        variance_s_b = np.dot(np.dot(self.__w_pca.T, s_b), self.__w_pca)
        variance_s_w = np.dot(np.dot(self.__w_pca.T, s_w), self.__w_pca)

        matrix = np.dot(np.linalg.inv(variance_s_w), variance_s_b)

        epsilon, values = np.linalg.eig(matrix)

        sorted_idx = np.argsort(epsilon)[::-1][:m]
        w_fld = values[:, sorted_idx]

        return np.real(np.dot(self.__w_pca, w_fld))

    def __get_projection(self, face):
        face_subspace = np.reshape(face, (56 * 46))
        projection = np.dot(self.__w.T, face_subspace - self.__mean_face)
        return projection  # weighted sum slide 7

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
            plt.imshow(self.__x[i], cmap='gray')
            plt.axis("off")
        plt.show()

        errors = []

        plt.figure()
        for i in range(n_img):
            reconstruction = self.get_reconstruction(self.__x[i])
            errors.append(self.__x[i] - reconstruction)
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
