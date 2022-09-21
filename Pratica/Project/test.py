import cv2

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(-1)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()


def imagenormalize(imagearr):
    facesarr = []
    for i in imagearr:
        pathimg = pathimage + i
        faces = cv2.imread(pathimg)  # Read image
        facesgray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        facerecognition = face_cascade.detectMultiScale(facesgray)

        eyes = eye_cascade.detectMultiScale(facesgray)  # (ex,ey,ew,eh)
        eyesrec = check_eyes(eyes, facerecognition)

        lefteyeCenter = (int(eyesrec[0][0] + eyesrec[0][2] / 2), int(eyesrec[0][1] + eyesrec[0][3] / 2), 1)
        righteyeCenter = (int(eyesrec[1][0] + eyesrec[1][2] / 2), int(eyesrec[1][1] + eyesrec[1][3] / 2), 1)

        xDiff = righteyeCenter[0] - lefteyeCenter[0]
        yDiff = righteyeCenter[1] - lefteyeCenter[1]
        angle = degrees(atan2(yDiff, xDiff))
        print("Original angle: " + str(angle))

        # cv2.line(faces, lefteyeCenter, righteyeCenter, (0, 255, 0), 1)

        image_center = tuple(np.array(faces.shape[1::-1]) / 2)

        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(faces, rot_mat, faces.shape[1::-1], flags=cv2.INTER_LINEAR)
        startpoint = np.asarray([lefteyeCenter, righteyeCenter])

        newpoint = np.dot(rot_mat, startpoint.T);

        newLefteyeCenter = (newpoint[0][0], newpoint[1][0])
        newRighteyeCenter = (newpoint[0][1], newpoint[1][1])

        newXDiff = newRighteyeCenter[0] - newLefteyeCenter[0]
        newYDiff = newRighteyeCenter[1] - newLefteyeCenter[1]

        newAngle = degrees(atan2(newYDiff, newXDiff))

        print("New angle: " + str(newAngle))

        nle = (int(newpoint[0][0]), int(newpoint[1][0]))
        nre = (int(newpoint[0][1]), int(newpoint[1][1]))

        eyedist = nre[0] - nle[0]
        scaling = 15 / eyedist

        nlescaled = [int(nle[0] * scaling), int(nle[1] * scaling)]
        nrescaled = [int(nre[0] * scaling), int(nre[1] * scaling)]

        newsize = (int(result.shape[1] * scaling), int(result.shape[0] * scaling))

        imgresized = cv2.resize(result, newsize, interpolation=cv2.INTER_NEAREST)

        x_t = 16 - nlescaled[0]
        y_t = 24 - nlescaled[1]
        translation_matrix = np.float32([[1, 0, x_t], [0, 1, y_t]])
        img_translation = cv2.warpAffine(imgresized, translation_matrix, (newsize[1], newsize[0]))

        img_translation_gray = cv2.cvtColor(img_translation, cv2.COLOR_BGR2GRAY)

        img_normalized = img_translation_gray[0:56, 0:46]

        np.append(facesarr, img_normalized)
    return facesarr
