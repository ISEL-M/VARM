import os
import random

import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt

from calibration import Calibration


class Markers:
    def __init__(self, calibration_data):
        self.__mtx = calibration_data['mtx']
        self.__dist = calibration_data['dist']
        self.__obj_points = calibration_data['obj_points']
        # self.__r_vecs = calibration_data['r_vecs']
        # self.__t_vecs = calibration_data['t_vecs']

        self.__dictionary = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.__dictionary, parameters)

        self.__marker_size = 0.5

        self.__load_objects()

    def __load_objects(self):
        self.__images = []
        for files in os.listdir("resources/virtual-objects/"):
            img_dir = os.path.join("resources/virtual-objects/", files)
            img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
            self.__images.append(img)

    def get_marker(self, idx=random.randint(1, 250), show=True, save_path="results/marker.png"):
        marker = aruco.generateImageMarker(self.__dictionary, idx, 200)

        cv2.imwrite(save_path, marker)

        if show:
            cv2.imshow("Marker", marker)
            cv2.waitKey(5000)

    def get_markers(self, n_markers=random.randint(1, 25), show=True, save_path="/results/marker.png"):
        # TODO
        new_markers = aruco.generateImageMarker(self.__dictionary, 23, 200)
        cv2.imshow("ok", new_markers)

        plt.figure()
        for i in range(12):
            plt.subplot(4, 4, i + 1)
            marker = aruco.generateImageMarker(self.__dictionary, i, 200)
            plt.imshow(marker, cmap='gray')
            plt.axis("off")
        plt.show()
        cv2.waitKey(5000)

    def detect_markers(self, frame, debug=False):
        # https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

        corners, ids, _ = self.detector.detectMarkers(frame)

        if debug:
            aruco.drawDetectedMarkers(frame, corners, ids)

        return corners, ids

    def draw(self, frame, corners, ids, debug=False):
        r_vecs, t_vecs, _ = aruco.estimatePoseSingleMarkers(corners, self.__marker_size*2, self.__mtx, self.__dist)

        idx = t_vecs.argsort(axis=0)[::-1, :, 2].squeeze()
        corners = np.asarray(corners)[idx]
        ids = ids[idx]
        r_vecs = r_vecs[idx]
        t_vecs = t_vecs[idx]

        for i in range(len(ids)):
            if not debug and ids[i] % 10 < len(self.__images):
                virtual_object = self.__images[(ids[i] % 10).squeeze()]
            else:
                virtual_object = None

            if 0 <= ids[i] < 10:
                self.draw_square(frame, corners[i].squeeze(), r_vecs[i], t_vecs[i], virtual_object)
            elif 10 <= ids[i] < 20:
                self.draw_box(frame, corners[i].squeeze(), r_vecs[i], t_vecs[i], virtual_object)

        return frame

    def draw_square(self, frame, corners, r_vec, t_vec, virtual_object):
        if virtual_object is None:
            aruco.drawDetectedMarkers(frame, [corners.reshape((1, 4, 2))], borderColor=(0, 0, 255))
            cv2.drawFrameAxes(frame, self.__mtx, self.__dist, r_vec, t_vec, .5)
        else:
            self.__add_virtual_object(frame, corners, virtual_object)

    def draw_box(self, frame, corners, r_vec, t_vec, virtual_object):
        new_corners = []
        n_corner = self.__3d_points((-1, 1), (1, 1), (1, 1), (-1, 1), (1, 1, 0, 0), r_vec, t_vec)  # north
        new_corners.append(n_corner)

        n_corner = self.__3d_points((1, 1), (1, -1), (1, -1), (1, 1), (1, 1, 0, 0), r_vec, t_vec)  # east
        new_corners.append(n_corner)

        n_corner = self.__3d_points((-1, 1), (-1, -1), (-1, -1), (-1, 1), (1, 1, 0, 0), r_vec, t_vec)  # west
        new_corners.append(n_corner)

        n_corner = self.__3d_points((-1, -1), (1, -1), (1, -1), (-1, -1), (1, 1, 0, 0), r_vec, t_vec)  # south
        new_corners.append(n_corner)

        n_corner = self.__3d_points((-1, 1), (1, 1), (1, -1), (-1, -1), (1, 1, 1, 1), r_vec, t_vec)  # top
        new_corners.append(n_corner)

        new_corners = np.asarray(new_corners).reshape((5, 1, 4, 2))

        r_vecs, t_vecs, _ = aruco.estimatePoseSingleMarkers(new_corners, self.__marker_size, self.__mtx, self.__dist)
        idx = t_vecs.argsort(axis=0)[::-1, :, 2].squeeze()
        new_corners = new_corners[idx]

        for i in range(len(new_corners)):
            if virtual_object is None:
                aruco.drawDetectedMarkers(frame, [corners.reshape((1, 4, 2))], borderColor=(0, 0, 0))
            else:
                self.__add_virtual_object(frame, new_corners[i], virtual_object)

    @staticmethod
    def __add_virtual_object(frame, corners, virtual_object):
        # https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

        height, width, _ = virtual_object.shape
        virtual_object_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        matrix, _ = cv2.findHomography(virtual_object_corners, corners)

        height, width, _ = frame.shape
        warped_virtual_object = cv2.warpPerspective(virtual_object, matrix, (width, height))

        cv2.fillConvexPoly(frame, corners.astype(int), (0, 0, 0))
        frame += warped_virtual_object

    def __3d_points(self, a, b, c, d, z, r_vec, t_vec):
        axis = np.array([[a[0] * self.__marker_size, a[1] * self.__marker_size, z[0]],
                         [b[0] * self.__marker_size, b[1] * self.__marker_size, z[1]],
                         [c[0] * self.__marker_size, c[1] * self.__marker_size, z[2]],
                         [d[0] * self.__marker_size, d[1] * self.__marker_size, z[3]]])
        points, _ = cv2.projectPoints(axis, r_vec, t_vec, self.__mtx, self.__dist)
        return points


if __name__ == "__main__":
    markers = Markers(Calibration('resources/calib.npz').data)
    markers.get_marker()
    markers.get_markers()
