import cv2
import dlib
import numpy as np


class FaceDetector:
    def __init__(self, debug=False, cuda=False):
        self.debug = debug
        self.cuda = cuda

        self.face_cascade = cv2.CascadeClassifier("models/haar_cascades/haar_cascade_frontal_face_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("models/haar_cascades/haar_cascade_eye.xml")

        proto_txt = 'models/dnn/deploy.prototxt.txt'
        caffe_model = 'models/dnn/res10_300x300_ssd_iter_140000.caffemodel'
        self.model = cv2.dnn.readNetFromCaffe(proto_txt, caffe_model)

        if self.cuda:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.hogDetector = dlib.get_frontal_face_detector()
        self.hogPredictor = dlib.shape_predictor("models/dlib/shape_predictor_68_face_landmarks.dat")

    def haar_cascade(self, frame):
        faces = []
        eyes = []

        detected_faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for detected_face in detected_faces:
            (face_x, face_y, face_w, face_h) = detected_face
            if self.debug:
                cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)

            face_frame = frame[face_y:face_y + int(face_h / 2), face_x:face_x + face_w]

            detected_eyes = self.eye_cascade.detectMultiScale(face_frame, 1.3, 5)
            if len(detected_eyes) < 2:
                continue

            pair = []
            for (eye_x, eye_y, eye_w, eye_h) in detected_eyes[0:2]:
                eye_x_center = face_x + eye_x + int(eye_w / 2)
                eye_y_center = face_y + eye_y + int(eye_h / 2)
                pair.append([eye_x_center, eye_y_center])

                if self.debug:
                    cv2.circle(frame, (eye_x_center, eye_y_center), 2, (0, 255, 0), 1)

            faces.append(detected_face)
            eyes.append(pair)

        return faces, eyes

    def dnn(self, frame, confidence_threshold=0.7):
        faces = []
        eyes = []

        h, w, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.model.setInput(blob)
        detections = self.model.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > confidence_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                detected_face = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (face_x, face_y, face_x_end, face_y_end) = np.asarray(detected_face, dtype=int)
                face = [face_x, face_y, face_x_end - face_x, face_y_end - face_y]
                if self.debug:
                    cv2.rectangle(frame, (face_x, face_y), (face_x_end, face_x_end), (255, 0, 0), 2)

                face_frame = frame[face[1]:face[1] + int(face[3] / 2), face[0]:face[0] + face[2]]

                detected_eyes = self.eye_cascade.detectMultiScale(face_frame, 1.3, 5)
                if len(detected_eyes) < 2:
                    continue

                pair = []
                for (eye_x, eye_y, eye_w, eye_h) in detected_eyes[0:2]:
                    eye_x_center = face_x + eye_x + int(eye_w / 2)
                    eye_y_center = face_y + eye_y + int(eye_h / 2)
                    pair.append([eye_x_center, eye_y_center])

                    if self.debug:
                        cv2.circle(frame, (eye_x_center, eye_y_center), 2, (0, 255, 0), 1)

                faces.append(face)
                eyes.append(pair)
        return faces, eyes

    def dlib_hog(self, frame):
        faces = []
        eyes = []

        detected_faces = self.hogDetector(frame, 1)

        for detected_face in detected_faces:
            start_x = detected_face.left()
            start_y = detected_face.top()
            end_x = detected_face.right()
            end_y = detected_face.bottom()
            face = [start_x, start_y, end_x - start_x, end_y - start_y]

            if self.debug:
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

            detected_face_landmarks = self.hogPredictor(frame, detected_face)

            x = y = idx = 0
            for n in range(36, 42):
                x += detected_face_landmarks.part(n).x
                y += detected_face_landmarks.part(n).y
                idx += 1
                # cv2.circle(frame, (x, y), 2, (0, 255, 0), 1)

            x = int(x / idx)
            y = int(y / idx)
            left_eye_center = [x, y]
            # cv2.circle(frame, left_eye_center, 2, (0, 255, 0), 1)

            x = y = idx = 0
            for n in range(42, 48):
                x += detected_face_landmarks.part(n).x
                y += detected_face_landmarks.part(n).y
                idx += 1
                # cv2.circle(frame, (x, y), 2, (0, 255, 0), 1)

            x = int(x / idx)
            y = int(y / idx)
            right_eye_center = [x, y]

            if self.debug:
                cv2.circle(frame, left_eye_center, 2, (0, 255, 0), 1)
                cv2.circle(frame, right_eye_center, 2, (0, 255, 0), 1)

            faces.append(face)
            eyes.append([left_eye_center, right_eye_center])
        return faces, eyes
