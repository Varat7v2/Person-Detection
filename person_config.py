#!/usr/bin/env python3

### FROZEN GRAPH --> PERSON DETECTION MODEL
FROZEN_GRAPH_PERSON = './models/faster_rcnn_restnet50_od.pb'
FROZEN_LABELS = './data/mscoco_label_map.pbtxt'
FROZEN_CLASSES = 90


### YOLO-V2 and YOLO-TINY MODEL PARAMETERS
YOLO_CONFIDENCE = 0.5	#reject boxes with confidence < 0.5
YOLO_THRESHOLD = 0.5	#to apply Non-Max Supression
YOLO_WEIGHTS = './models/yolo_models/yolov3-416.weights'
YOLO_CONFIG = './models/yolo_models/yolov3-416.cfg'
YOLO_LABELS = './models/yolo_models/coco-labels'

### TFLITE MODEL PARAMETERS
TFLITE_MODEL = './models/ssd_mobilenetv2_od.tflite'
TFLITE_THRESHOLD = 0.4

#CAMERA AND VIDEO PARAMETERS
RUN_CAMERA = False
WRITE_VIDEO = True
myVIDEO_STREAM = False
CAMERA_ID = 0
VIDEO_INPUT = './videos/cctv.mp4'
VIDEO_OUTPUT = './person_detection.avi'