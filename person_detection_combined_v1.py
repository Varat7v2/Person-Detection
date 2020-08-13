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

FLAGS = list()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    frame_count = 0
    t_cam = 0

    parser.add_argument('-m', '--model',
        type=str,
        default='tflite_model',
        help='Choose the model for inference: 1) tflite_model, 2)frozen_model, 3)yolo_model')

    ### TFLITE PARAMETERS   PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
    parser.add_argument('-tm', '--tflite_path',
        type=str,
        default='./models/ssd_mobilenetv2_od.tflite',
        help='Path to the TFLITE model.')

    ### FROZEN GRAPH PARAMETERS
    parser.add_argument('-fm', '--frozen_path',
        type=str,
        default='./models/ssd_resnet_od.pb',
        help='Path to the FROZEN INFERENCE GRAPH model.')

    parser.add_argument('-fl', '--frozen_label',
        type=str,
        default='./data/mscoco_label_map.pbtxt',
        help='Path to the MSCOCO label.')

    parser.add_argument('-fc', '--frozen_class',
        type=int,
        default=90,
        help='Number of classes.')

    ### YOLO PARAMETERS
    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.8,
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

    # # MODEL PATHS and PARAMETERS - TODO: Put these into separate config file
    # if FLAGS.model == 'tflite':
    #     model_path = './models/detect.tflite'
    # if FLAGS.model == 'frozen_graph':
    #     model_path = './models/fg_faster_rcnn_restnet50.pb'
    #     frozen_label = './data/mscoco_label_map.pbtxt'
    #     frozen_class = 90
    # if FLAGS.model == 'yolo':
    #     model_weights = './models/yolo_models/yolov3.weights'
    #     model_config = './models/yolo_models/yolov3.cfg'
    #     nms_threshold = 0.3     ### threshold for when to apply Non-Max Suppresion
    #     model_confidence = 0.5  ### rejects boundaries less with confidence < 0.5
    #     output_labels = './models/yolo_models/coco-labels'

    ### FROZEN GRAPH MODEL INITIALIZATION
    label_map = label_map_util.load_labelmap(FLAGS.frozen_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=FLAGS.frozen_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    ### YOLO MODEL INITIALIZATION
    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(FLAGS.labels).read().strip().split('\n')
    
    ### PERSON DETECTOR objects
    tflite_detector = TFLITE_INFERENCE(FLAGS)
    frozen_detector = FROZEN_GRAPH_INFERENCE(FLAGS)
    yolo_detector = YOLO_INFERENCE(FLAGS)
    parser = argparse.ArgumentParser()

    # source = 0
    source = 'videos/cctv.mp4'

    # cap = webcamVideoStream(source).start()
    cap = cv2.VideoCapture(source)

    _, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]

    vid_writer = cv2.VideoWriter('output_video.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'), 22, (frame_width, frame_height))

    frame_count = 0
    tt_opencvDnn = 0
    print("process started ...")

    while(True):
        t = time.time()
        hasFrame, frame = cap.read()

        if hasFrame is False:
            break

        frame_count += 1

        # im_height, im_width = frame.shape[:2]
        im_height, im_width, _ = frame.shape

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
        vid_writer.write(frame)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(1)
        if k == 27:
            break

    print("process ended.")
    cv2.destroyAllWindows()
    cap.release()
    vid_writer.release()
