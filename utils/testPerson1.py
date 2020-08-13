import numpy as np
import tensorflow as tf
import cv2
import sys
import time

from PersonDets1 import ObjectDetectorLite
from myVideoStream import webcamVideoStream

def rotateImage(image, angle):
    row, col, channel = image.shape
    # print(row, col, channel)
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    # new_image = cv2.resize(new_image, (480, 640))
    return new_image

if __name__ == '__main__':
    detector = ObjectDetectorLite()
    vs = webcamVideoStream(src=0).start()
    # cap = cv2.VideoCapture(0)

    frame_count = 0
    tt_opencvDnn = 0
    oldPoints = []

    # Lucas kanade params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    count = 0
    # _, frame = cap.read()
    frame = vs.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame, bottompts = detector.detect(frame, 0.5)

    while (True):
        t = time.time()
        # _, frame = cap.read()
        frame = vs.read()
        # frame = cv2.resize(frame, (600, 600))
        frame = rotateImage(frame, 90)

        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frame = cv2.flip(frame, 1)

        if (count < 1):
            frame, bottompts = detector.detect(frame, 0.5)
            x, y = bottompts
            # for x, y in bottompts:
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        else:
            newPoints, status, error = cv2.calcOpticalFlowPyrLK(old_gray,gray_frame,oldPoints,None, **lk_params)
            x, y = newPoints.ravel()
            old_gray = gray_frame.copy()
            oldPoints = newPoints
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        tdiff = time.time() - t
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        # print("FPS: {:.2f}".format(1/tdiff))

        label = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("Face Detection Comparison", frame)
        count += 1

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    vs.stop()
