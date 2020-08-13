import numpy as np
import tensorflow as tf
import argparse
import cv2
import time

from myTFLITE import TFLITE_INFERENCE
from myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE
from myYOLO import YOLO_INFERENCE

from myVideoStream import webcamVideoStream
from utils import label_map_util
import person_config as config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    frame_count = 0
    t_cam = 0

    parser.add_argument('-m', '--model',
        type=str,
        default='tflite_model',
        help='Choose the model for inference: 1) tflite_model, 2)frozen_model, 3)yolo_model')

    FLAGS, unparsed = parser.parse_known_args()

    ### FROZEN GRAPH MODEL INITIALIZATION
    label_map = label_map_util.load_labelmap(config.FROZEN_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=config.FROZEN_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    ### YOLO MODEL INITIALIZATION
    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(config.YOLO_CONFIG, config.YOLO_WEIGHTS)
    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(config.YOLO_LABELS).read().strip().split('\n')
    
    ### PERSON DETECTOR objects
    tflite_detector = TFLITE_INFERENCE(config.TFLITE_MODEL)
    frozen_detector = FROZEN_GRAPH_INFERENCE(config.FROZEN_GRAPH_PERSON)
    yolo_detector = YOLO_INFERENCE(config.YOLO_CONFIDENCE, config.YOLO_THRESHOLD)

    if config.RUN_CAMERA:
        source = config.CAMERA_ID
    else:
        source = config.VIDEO_INPUT

    if config.myVIDEO_STREAM:
        cap = webcamVideoStream(source).start()
    else:
        cap = cv2.VideoCapture(source)

    if config.WRITE_VIDEO:
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, 30, (frame_width, frame_height))
        frame_index = -1

    frame_count = 0
    tt_opencvDnn = 0
    print("process started ...")

    while(True):
        t = time.time()
        hasFrame, frame = cap.read()

        if hasFrame is False:
            break

        frame_count += 1

        im_height, im_width = frame.shape[:2]

        if FLAGS.model == 'tflite_model':
            ### RUN --> TFLITE MODEL
            frame = tflite_detector.detect(frame)
        if FLAGS.model == 'frozen_model':
            ### RUN --> FROZEN INFERENCE GRAPH 
            frame, boxes, scores, classes, num_detections = frozen_detector.run_frozen_graph(frame, 
                im_width, im_height)
        if FLAGS.model == 'yolo_model':
            ### RUN --> YOLO MODEL
            frame, boxes, confidences, classids, idxs = yolo_detector.run_yolo(frame, net, 
                layer_names, im_width, im_height, labels)
        # else:
            # print("Choose any one of the model: 1) tflite_model, 2) frozen_model, or 3) yolo_model")

        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        fps = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(frame, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", frame)

        if config.WRITE_VIDEO:
            out.write(frame)
            frame_index = frame_index + 1

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    if config.WRITE_VIDEO:
        out.release()
    cv2.destroyAllWindows()
