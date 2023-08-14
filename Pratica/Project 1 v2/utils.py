import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_images_from_folders(directory="resources/faces"):
    x = []
    y = []

    main_dir = os.listdir(directory)
    for folder in main_dir:
        if folder == "new":
            continue

        folder_dir = os.path.join(directory, folder)
        for filename in os.listdir(folder_dir):

            img_dir = os.path.join(folder_dir, filename)
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                x.append(img)
                y.append(folder)

    return np.asarray(x), np.asarray(y)


def normalize_face(frame, eyes):
    h, w, channels = frame.shape
    frame_center = (w // 2, h // 2)

    diff_x_eye = eyes[1][0] - eyes[0][0]
    diff_y_eye = eyes[1][1] - eyes[0][1]
    angle = math.degrees(math.atan2(diff_y_eye, diff_x_eye))

    rotation_matrix = cv2.getRotationMatrix2D(frame_center, angle, 1.)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))

    rotated_right_eye = np.matmul(rotation_matrix, eyes[0] + [1])
    rotated_left_eye = np.matmul(rotation_matrix, eyes[1] + [1])

    # cv2.circle(rotated_frame, (int(rotated_right_eye[0]), int(rotated_right_eye[1])), 2, (0, 255, 0), 1)
    # cv2.circle(rotated_frame, (int(rotated_left_eye[0]), int(rotated_left_eye[1])), 2, (0, 255, 0), 1)

    wanted_distance_between_eyes = 15
    actual_distance_between_eyes = rotated_left_eye[0] - rotated_right_eye[0]
    # actual_distance_between_eyes2 = np.sqrt(np.sum((rotated_right_eye - rotated_left_eye) ** 2))

    scale = wanted_distance_between_eyes / actual_distance_between_eyes

    resized_rotated_frame = cv2.resize(rotated_frame, (int(w * scale), int(h * scale)))

    start = np.asarray(rotated_right_eye * scale, dtype=int) - [16, 24]
    end = np.asarray(rotated_right_eye * scale, dtype=int) + [16 + 14, 56 - 24]

    ok = end - start

    normalized_face = resized_rotated_frame[start[1]:end[1], start[0]:end[0]]

    gray_normalized_face = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2GRAY)
    return gray_normalized_face, angle


def add_gaara_mark(frame, face, eyes, angle):
    (_, _, w, h) = face
    (eyes_x, eyes_y) = eyes[0]

    mark = cv2.imread('resources/objects/gaara.png')
    # mark = cv2.rotate(mark, int(angle))

    rows, cols, _ = mark.shape
    scale = (abs(eyes[0][0] - eyes[1][0]) - 50) / rows
    mark = cv2.resize(mark, (int(cols * scale), int(rows * scale)))
    rows, cols, _ = mark.shape

    a = eyes_y - 100
    b = eyes_x - 40

    roi = frame[a:rows + a, b:cols + b]
    grey_band = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(grey_band, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    cv2.imshow("mask2", img1_bg)
    img2_fg = cv2.bitwise_and(mark, mark, mask=mask)
    cv2.imshow("mask", img2_fg)

    dst = cv2.add(img1_bg, img2_fg)
    cv2.imshow("mask2", dst)

    frame[a:rows + a, b:cols + b] = dst


def get_mean_faces():
    x, y = load_images_from_folders()

    unique_ys = np.unique(y)

    mean_face = np.mean(x, axis=0)
    mean_faces = [np.mean(x[uni_y == y], axis=0) for uni_y in unique_ys]

    plt.figure()
    plt.imshow(mean_face, cmap='gray')
    plt.axis("off")
    plt.title('mean faces')
    plt.show()

    n = len(unique_ys)
    size = int(np.ceil(np.sqrt(n)))

    plt.figure()
    for i in range(n):
        plt.subplot(size, size, i + 1)
        plt.imshow(mean_faces[i - 1], cmap='gray')
        plt.axis("off")
        plt.title(unique_ys[i - 1] + ' mean face')
    plt.show()

# get_mean_faces()
