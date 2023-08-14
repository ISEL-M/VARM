import glob
import os

import cv2
import numpy as np


class Calibration:
    def __init__(self, load_path=None, debug=False):
        self._debug = debug

        if load_path is not None and os.path.isfile(load_path):
            self.__load_calib_values(load_path)
        else:
            self.__calculate_calib_values()

    def __calculate_calib_values(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        gray = None
        images = glob.glob('resources/calib-img/*.jpg')
        for img in images:
            img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

                if self._debug:
                    cv2.imshow('img', img)
                    cv2.waitKey(5000)

        ret, self.__mtx, self.__dist, r_vecs, t_vecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        np.savez("resources/calib.npz",
                 mtx=self.__mtx,
                 dist=self.__dist,
                 r_vecs=r_vecs,
                 t_vecs=t_vecs,
                 obj_points=obj_points)

        if self._debug:
            mean_error = 0
            for i in range(len(obj_points)):
                img_points2, _ = cv2.projectPoints(obj_points[i], r_vecs[i], t_vecs[i], self.__mtx, self.__dist)
                error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                mean_error += error
            print("total error: {}".format(mean_error / len(obj_points)))

    def __load_calib_values(self, path):
        self.data = np.load(path)
        self.__mtx = self.data['mtx']
        self.__dist = self.data['dist']
        self.__r_vecs = self.data['r_vecs']
        self.__t_vecs = self.data['t_vecs']

    def calibrate(self, frame, remap_instead_of_undistort=False):
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.__mtx, self.__dist, (w, h), 1, (w, h))

        # undistort
        if remap_instead_of_undistort:
            map_x, mapy = cv2.initUndistortRectifyMap(self.__mtx, self.__dist, None, new_camera_mtx, (w, h), 5)
            dst = cv2.remap(frame, map_x, mapy, cv2.INTER_LINEAR)
        else:
            dst = cv2.undistort(frame, self.__mtx, self.__dist, None, new_camera_mtx)

        # crop the image
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]
