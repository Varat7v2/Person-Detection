import numpy as np
import tensorflow as tf
import cv2

from object_detector_detection_api import ObjectDetectorDetectionAPI, PATH_TO_LABELS, NUM_CLASSES

class TFLITE_INFERENCE(ObjectDetectorDetectionAPI):
	def __init__(self, tflite_model):

		# Load the model
		model_path = tflite_model

		# Load label_map
		self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

		# Load TFLite model and allocate tensors.
		self.interpreter = tf.lite.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()

		# Get input and output tensors.
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def detect(self, image, threshold=0.4):
	    box0 = []
	    box1 = []

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
	    for obj in result:
	        if(obj[3] == 'person'):
	            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
	            cv2.putText(image, '{}: {:.3f}'.format(obj[3], obj[2]),(obj[0][0], obj[0][1] - 5),
	                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
	           
	            box0.append(obj[0])
	            box1.append(obj[1])

	    ### NEED TO CUSTOMIZE LATER
	    # boxes = np.squeeze(boxes)
	    # scores = np.squeeze(scores)
	    # classes = np.squeeze(classes).astype(np.int32)

	    # for score, box, name in zip(scores, boxes, classes):

	    # 	if name == 0 and score > 0.7:
	    #         # ymin, xmin, ymax, xmax = box
	    #         left = int((box[1]*127.5+1.0)*300*im_width)
	    #         top = int((box[0]*127.5+1.0)*300*im_height)
	    #         right = int((box[3]*127.5+1.0)*300*im_width)
	    #         bottom = int((box[2]*127.5+1.0)*300*im_height)

	    #         cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1, 8)

	    return image