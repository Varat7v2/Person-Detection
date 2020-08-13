import numpy as np
import tensorflow as tf
import cv2
import sys
import time

from object_detector_detection_api import ObjectDetectorDetectionAPI, PATH_TO_LABELS, NUM_CLASSES


class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path='./models/ssd_mobilenetv2_od.tflite'):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        center_pt = (0,0)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(frame, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = (2.0 / 255.0) * frame - 1.0
        frame = frame.astype('float32')

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # print(boxes)

        # Find detected boxes coordinates
        result = self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)
        bottom_mids = list()
        bottom_mid = (0,0)

        for obj in result:
            if(obj[3] == 'person'):
                # obj[0] and [1] --> rect coordinates; obj[3] --> class; obj[2]--> confidence
                width = obj[1][0] - obj[0][0]
                height = obj[1][1] - obj[0][1]
                # center_pt = (obj[0][0] + int(width/2), obj[0][1] + int(height/2))
                bottom_mid = (obj[0][0] + int(width/2), obj[0][1] + height)
                left = obj[0][0]
                top = obj[0][1]
                right = obj[1][0]
                bottom = obj[1][1]
                confidence = obj[2]
                label = obj[3]

                cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
                cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                            (obj[0][0], obj[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                bottom_mids.append(bottom_mid)
        
        # for (i,j) in bottom_mids:
        #         	print(i, j)

        return image, bottom_mids