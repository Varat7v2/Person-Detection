import numpy as np
import tensorflow as tf
import argparse
import cv2
import time

from myTFLITE import TFLITE_INFERENCE

FLAGS = list()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    frame_count = 0
    t_cam = 0

    parser.add_argument('-m', '--tflite_model',
        type=str,
        default='./models/ssd_mobilenetv2_od.tflite',
        help='Path to the model.')

    FLAGS, unparsed = parser.parse_known_args()

    detector = TFLITE_INFERENCE(FLAGS)
    parser = argparse.ArgumentParser()

    # source = 0
    source = 'videos/shoppingMall.mp4'

    cap = cv2.VideoCapture(source)
    # vs = webcamVideoStream(source).start()

    frame_count = 0
    tt_opencvDnn = 0

    while(True):
        t = time.time()
        hasFrame, frame = cap.read()

        frame_count += 1

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = detector.detect(frame, 0.4)

        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        label = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", frame)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
    # vs.stop()