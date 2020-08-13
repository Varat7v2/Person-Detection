#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2

from object_detector_detection_api import ObjectDetectorDetectionAPI, PATH_TO_LABELS, NUM_CLASSES

class TFLITE_INFERENCE(ObjectDetectorDetectionAPI):
	def __init__(self, tf_model):

		# Load the model
		model_path = tf_model

		# Load label_map
		self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

		# Load TFLite model and allocate tensors.
		self.interpreter = tf.lite.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()

		# Get input and output tensors.
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def run_model(self, image, threshold):
	    box0 = list()
	    box1 = list()

	    im_width, im_height = image.shape[:2]

	    # Resize and normalize image for network input
	    frame = cv2.resize(image, (300, 300))
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

	    # Find detected boxes coordinates
	    result =  self._boxes_coordinates(image,
	                        np.squeeze(boxes[0]),
	                        np.squeeze(classes[0]+1).astype(np.int32),
	                        np.squeeze(scores[0]),
	                        min_score_thresh=threshold)

	    persons = list()
	    idx = 0
	    
	    for obj in result:
	        if (obj[3] == 'person'):
	            # obj[0] and [1] --> rect coordinates; obj[3] --> class; obj[2]--> confidence
	            width = obj[1][0] - obj[0][0]
	            height = obj[1][1] - obj[0][1]
	            # center_pt = (obj[0][0] + int(width/2), obj[0][1] + int(height/2))
	            bottom_mid = (obj[0][0] + int(width / 2), obj[0][1] + height)
	            left = obj[0][0]
	            top = obj[0][1]
	            right = obj[1][0]
	            bottom = obj[1][1]
	            confidence = obj[2]
	            label = obj[3]

	            mydict = {
	                "width": width,
	                "height": height,
	                "left": left,
	                "right": right,
	                "top": top,
	                "bottom": bottom,
	                "confidence": confidence,
	                "label": label + str(idx),
	                "bottom_mid": bottom_mid,
	                "model_type": 'TFLITE'
	                }
	            persons.append(mydict)
	            idx += 1

	    return persons