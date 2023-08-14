import cv2
from calibration import Calibration
from markers import Markers
'resources/calib.npz'

# camera_calibration = Calibration(-1)
camera_calibration = Calibration('resources/calib.npz')
mrk = Markers(camera_calibration.data)

vid = cv2.VideoCapture(0)
while True:
    key = cv2.waitKey(50)
    if key == ord('q'):
        break

    ret, frame = vid.read()

    # cv2.imshow("original", frame)

    undistorted_frame = camera_calibration.calibrate(frame=frame)
    # undistorted_frame = camera_calibration.calibrate(frame=frame, remap_instead_of_undistort=True)
    # cv2.imshow("undistort", undistorted_frame)
    corners, ids = mrk.detect_markers(undistorted_frame, debug=True)

    if ids is None:
        cv2.imshow("ok", undistorted_frame)
        cv2.waitKey(30)
        continue

    mrk.draw(frame=undistorted_frame, corners=corners, ids=ids, debug=False)
    cv2.imshow("ok", undistorted_frame)


vid.release()
cv2.destroyAllWindows()
