import sys
import time
import numpy as np
import tensorflow as tf
import argparse
import cv2

from utils import label_map_util
from myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE

FLAGS = list()

writeVideo_flag = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--frozen_model',
        type=str,
        default='./models/faster_rcnn_restnet50_od.pb',
        help='Path to the model.')

    FLAGS, unparsed = parser.parse_known_args()

    detector = FROZEN_GRAPH_INFERENCE(FLAGS)
    parser = argparse.ArgumentParser()

    # source = 0
    source = 'videos/shoppingMall.mp4'
    cap = cv2.VideoCapture(source)
    im_width = int(cap.get(3))
    im_height = int(cap.get(4))

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
    NUM_CLASSES = 90
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter('person_detection_frozenGraph.avi', fourcc, 30, (im_width, im_height))
        frame_index = -1

    while True:
        t1 = time.time()
        ret, frame = cap.read()

        if ret == 0:
            break

        im_height, im_width, im_channel = frame.shape
        frame = cv2.flip(frame, 1)

        frame, boxes, scores, classes, num_detections = detector.run_frozen_graph(frame, im_width, im_height)

        t2 = time.time() - t1
        fps = 1 / t2

        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("FACE DETECTION USING FROZEN GRAPH", frame)

        if writeVideo_flag:
            video_output.write(frame)
            frame_index = frame_index + 1
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    if writeVideo_flag:
        video_output.release()
    cv2.destroyAllWindows()
