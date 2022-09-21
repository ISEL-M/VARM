import math

import cv2
import numpy as np


def normalize_face(img, faces, eyesPos):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(faces)):
        if len(eyesPos[i]) == 2:
            xDiff = eyesPos[i][0][0] - eyesPos[i][1][0]
            yDiff = eyesPos[i][0][1] - eyesPos[i][1][1]
            angle = math.degrees(math.atan2(yDiff, xDiff))
            print("Original angle: " + str(angle))

            #cv2.line(faces, eyes_pos[i][0], eyes_pos[i][1], (0, 255, 0), 1)
            center = tuple(np.array(img.shape[1::-1]) / 2)

            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
            startpoint = np.asarray([eyesPos[i][0], eyesPos[i][1]])
            newpoint = np.dot(rot_mat, startpoint)


            leftEye = (newpoint[0][0], newpoint[1][0])
            rightEye = (newpoint[0][1], newpoint[1][1])

            newXDiff = rightEye[0] - leftEye[0]
            newYDiff = rightEye[1] - leftEye[1]

            newAngle = math.degrees(math.atan2(newYDiff, newXDiff))

            print("New angle: " + str(newAngle))
            nle = (int(newpoint[0][0]), int(newpoint[1][0]))
            nre = (int(newpoint[0][1]), int(newpoint[1][1]))

            eyedist = nre[0] - nle[0]
            scaling = 15 / eyedist

            nlescaled = [int(nle[0] * scaling), int(nle[1] * scaling)]
            nrescaled = [int(nre[0] * scaling), int(nre[1] * scaling)]

            newsize = (int(result.shape[1] * scaling), int(result.shape[0] * scaling))

            imgresized = cv2.resize(result, newsize)

            x_t = 16 - nlescaled[0]
            y_t = 24 - nlescaled[1]
            translation_matrix = np.float32([[1, 0, x_t], [0, 1, y_t]])
            img_translation = cv2.warpAffine(imgresized, translation_matrix, (newsize[1], newsize[0]))

            img_translation_gray = cv2.cvtColor(img_translation, cv2.COLOR_BGR2GRAY)

            img_normalized = img_translation_gray[0:56, 0:46]

            cv2.imshow("new face", frame_faces_eyes)
            #np.append(facesarr, img_normalized)

    return


def cascade(img):
    classifier_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    classifier_eyes = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
    faces = classifier_face.detectMultiScale(img)  # result

    eyes_pos = []
    # to draw faces on image
    for result_face in faces:
        x, y, w, h = result_face
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

        face = img[y:y1, x:x1]
        eyes = classifier_eyes.detectMultiScale(face)

        result_eyes = []
        for result_eye in eyes:
            x_eye, y_eye, w, h = result_eye
            result_eyes.append((x + w/2, y + h/2))

            x1_eye, y1_eye = x + w, y + h
            cv2.rectangle(img, (x + x_eye, y + y_eye), (x_eye + x1_eye, y_eye + y1_eye), (0, 0, 255), 2)

        eyes_pos.append(result_eyes)

    return img, faces, eyes_pos


def dnn(img):
    import cv2
    import numpy as np
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    # to draw faces on image
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

    cv2.imshow("ok", img)
    cv2.waitKey(0)


vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    #frame = cv2.imread("ok.jpg")
    frame_faces_eyes, faces, eyes_pos = cascade(frame)
    cv2.imshow("Frame", frame_faces_eyes)

    normalize_face(frame, faces, eyes_pos)
    #cv2.imshow("Normalized Frame", normazed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

normalize_face(frame, faces, eyes_pos)
vid.release()
cv2.destroyAllWindows()
