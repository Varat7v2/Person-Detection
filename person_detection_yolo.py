import numpy as np
import argparse
import cv2
import subprocess
import time
import os
from myYOLO import YOLO_INFERENCE
from myVideoStream import webcamVideoStream

FLAGS = list()

writeVideo_flag = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    frame_count = 0
    t_cam = 0

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./models/yolo_models/yolov3.weights',
        help='Path to the file which contains the weights \
                for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./models/yolo_models/yolov3.cfg',
        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./models/yolo_models/coco-labels',
        help='Path to the file having the \
                    labels in a new-line seperated way.')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    yolo = YOLO_INFERENCE(FLAGS)

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    labels = open(FLAGS.labels).read().strip().split('\n')

    source = 'videos/shoppingMall.mp4'
    
    cap = cv2.VideoCapture(source)
    # vs = webcamVideoStream(source).start()
    im_width = int(cap.get(3))
    im_height = int(cap.get(4))

    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter('person_detection_frozenGraph.avi', fourcc, 30, (im_width, im_height))
        frame_index = -1

    while True:
        t1 = time.time()
        _, frame = cap.read()
        # frame = vs.read()
        frame_count += 1

        height, width, _ = frame.shape

        frame, boxes, confidences, classids, idxs = yolo.run_yolo(frame, net, 
            layer_names, width, height, labels)

        t_cam += time.time() - t1
        fps_cam = (frame_count/t_cam)

        cv2.putText(frame, "FPS: {:.2f}".format(fps_cam), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 
            1.4, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('webcam', frame)

        if writeVideo_flag:
            video_output.write(frame)
            frame_index = frame_index + 1

        if frame_count == 1:
            t_cam = 0

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    if writeVideo_flag:
        video_output.release()
    # vs.stop()
    cv2.destroyAllWindows()
