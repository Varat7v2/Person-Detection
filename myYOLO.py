#!/usr/bin/env python3

import numpy as np
import argparse
import cv2
import subprocess
import time
import os

class YOLO_INFERENCE:

    def __init__(self, confidence, threshold):
        # self.config_path = FLAGS.config
        # self.weights_path = FLAGS.weights
        # self.labels = open(FLAGS.labels).read().strip().split('\n')

        self.confidence = confidence
        self.threshold = threshold

    def draw_labels_and_boxes(self, img, boxes, confidences, classids, idxs, labels):

        persons = []
        idx = 0

        for obj, score, box in zip(classids, confidences, boxes):
            if obj == 0 and score > 0.7:

                # Get the bounding box coordinates
                x, y = box[0], box[1]
                w, h = box[2], box[3]

                left, top, right, bottom = x, y, x+w, y+h

                # Draw the bounding box rectangle and label on the image
                # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                # text = "{}: {:4f}".format(labels[0], score)
                # cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                width = right - left
                height = bottom - top
                bottom_mid = (left + int(width / 2), top + height)
                confidence = score
                label = labels[0]

                mydict = {
                    "width": width,
                    "height": height,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "confidence": confidence,
                    "label": None,
                    "bottom_mid": bottom_mid,
                    "model_type": 'YOLO'
                    }
                persons.append(mydict)
                idx += 1

        return persons

    def generate_boxes_confidences_classids(self, outs, height, width, tconf):

        boxes = []
        confidences = []
        classids = []

        for out in outs:
            for detection in out:
                # Get the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classid = np.argmax(scores)
                confidence = scores[classid]
                
                # Consider only the predictions that are above a certain confidence level
                if confidence > tconf:
                    # TODO Check detection
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, bwidth, bheight = box.astype('int')

                    # Using the center x, y coordinates to derive the top
                    # and the left corner of the bounding box
                    left = int(centerX - (bwidth / 2))
                    top = int(centerY - (bheight / 2))
                    # right = int(centerX + (bwidth / 2))
                    # bottom = int(centerY + (bwidth / 2))

                    # Append to list
                    boxes.append([left, top, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

        return boxes, confidences, classids

    def run_model(self, img, net, layer_names, width, height, labels):

        # Contructing a blob from the input image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector 
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        print("YOLO model inference time: ", (end-start), "seconds")

        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = self.generate_boxes_confidences_classids(outs, height, 
            width, self.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
            
        # Draw labels and boxes on the image
        persons = self.draw_labels_and_boxes(img, boxes, confidences, classids, idxs, labels)

        return persons