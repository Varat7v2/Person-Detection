import numpy as np
import tensorflow as tf
import cv2
import sys
import time
from run import Reid
from myVideoStream import webcamVideoStream

from object_detector_detection_api import ObjectDetectorDetectionAPI, PATH_TO_LABELS, NUM_CLASSES

class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path='detect.tflite'):
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
        self.count = 0

    def detect(self, image, threshold=0.1):
        box0 = []
        box1 = []
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

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
                if self.count == 0:
                    cv2.imwrite('./images/img1.jpg', image[obj[0][1]:obj[1][1], obj[0][0]:obj[1][0]])
                    self.count += 1

                cv2.imwrite('./images/img2.jpg', image[obj[0][1]:obj[1][1], obj[0][0]:obj[1][0]])
                # print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.format(obj[0], obj[1], obj[3], obj[2]))
                # obj[0] and [1] --> rect coordinates; obj[3] --> class; obj[2]--> confidence
                cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
                cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),(obj[0][0], obj[0][1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
               
                box0.append(obj[0])
                box1.append(obj[1])

        return image, box0, box1

if __name__ == '__main__':
    detector = ObjectDetectorLite()
    source = 0
    # source = '/home/varat/myPERSON_RE_ID/dataset_videos/video2.mp4'
    reid = Reid()
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    # vs = webcamVideoStream(source).start()

    frame_count = 0
    tt_opencvDnn = 0

    while(True):
        t = time.time()
        hasFrame, frame = cap.read()
        # frame = vs.read()
        # if not hasFrame:
        #     break
        frame_count += 1

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, X, Y = detector.detect(frame, 0.4)

        # ret = reid.compare('./images/img1.jpg', './images/img2.jpg')
        # cv2.putText(frame, str(ret), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)

        # img1 = cv2.imread('./images/img1')
        # img2 = cv2.imread('./images/img2')
        
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        label = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", frame)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
    # vs.stop()