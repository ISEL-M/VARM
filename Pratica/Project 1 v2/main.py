from eigen_faces import EigenFaces
from face_detection import FaceDetector
from fisher_faces import FisherFaces
from utils import *

face_detection = FaceDetector(debug=False, cuda=True)

eig_faces = EigenFaces(m=20)
fisher_faces = FisherFaces(m=20)

# eig_faces.get_results()
# fisher_faces.get_results()


vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()

    # faces, eyes = face_detection.haar_cascade(frame)
    faces, eyes = face_detection.dnn(frame)
    # faces, eyes = face_detection.dlib_hog(frame)

    for idx in range(len(faces)):
        normalized_face, angle = normalize_face(frame, eyes[idx])
        # TODO save normalized_face
        if not normalized_face.shape == (56, 46):
            continue

        prediction = eig_faces.get_prediction(normalized_face)
        # prediction = fisher_faces.get_prediction(normalized_face)

        print(prediction)

        if prediction == 'mihail':
            add_gaara_mark(frame=frame, face=faces[idx], eyes=eyes[idx], angle=angle)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# pedro.mendes.jorge@isel.pt
