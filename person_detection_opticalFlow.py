import time
import cv2
import numpy as np

from myVideoStream import webcamVideoStream
from myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE

def findEuclideanDist(A, B):
    arow, acol = A.shape
    brow, bcol = B.shape

    for i in range(arow):
        for j in range(brow):
            mydist[i,j] = A
            
if __name__ == '__main__':
    frozen_detector = FROZEN_GRAPH_INFERENCE(config.FROZEN_GRAPH_PERSON)

    if config.RUN_CAMERA:
        source = config.CAMERA_ID
    else:
        source = config.VIDEO_INPUT

    if config.myVIDEO_STREAM:
        cap = webcamVideoStream(source).start()
    else:
        cap = cv2.VideoCapture(source)

    frame_count = 0
    tt_opencvDnn = 0
    x, y = (0,0)

    # Lucas kanade params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    count = 0
    optical_num = 5
    max_dist = list()

    while(True):
        t1 = time.time()
        ret, frame  = cap.read()
        frame_count += 1
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_height, im_width = frame.shape[:2]

        # if (count % optical_num == 0):
        if True:
            oldPoints = list()
            widths = list()
            heights = list()
            frame, boxes, scores, classes, num_detections = frozen_detector.run_frozen_graph(frame, im_width, im_height)
            oldPoints = np.asarray(bottompts, dtype=np.float32)
            oldPoints_dist = oldPoints.copy()
            for idx, df in enumerate(persons):
                cv2.rectangle(frame, (df['left'], df['top']), (df['right'], df['bottom']), (0, 255, 0), 2)
                cv2.putText(frame, '{}: {:.2f}'.format(df['label'], df['confidence']),
                            (df['left'], df['top'] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                widths.append(df['width'])
                heights.append(df['height'])

        else:
            newPoints = []
            if(len(oldPoints) > 0):
                for idx, oldPoint in enumerate(oldPoints):
                    oldPoint = oldPoint.reshape((1,2))
                    newPoint, status, error = cv2.calcOpticalFlowPyrLK(old_gray,gray_new, oldPoint, None, **lk_params)
                    myleft = int(newPoint.ravel()[0] - widths[idx]/2)
                    mytop = int(newPoint.ravel()[1] - heights[idx])
                    myright = int(newPoint.ravel()[0] + widths[idx]/2)
                    mybottom = int(newPoint.ravel()[1])
                    cv2.rectangle(frame, (myleft, mytop),(myright, mybottom), (0, 0, 255), 2)

                    newPoints.append(newPoint)

                if(count % (optical_num-1) == 0):
                    newPoints = np.array(newPoints, dtype=np.float32)
                    newPoints = newPoints[:, 0, :]

                    if (oldPoints_dist.ndim == 3):
                        oldPoints_dist = oldPoints_dist[:, 0, :]

                    print('Euclidean Distances: ')
                    euclidean_dist = np.linalg.norm((oldPoints_dist - newPoints), axis=1)
                    max_dist.append(np.amax(euclidean_dist))
                    print(np.amax(max_dist))

                oldPoints = newPoints.copy()

        old_gray = gray_new.copy()


        tt_opencvDnn += time.time() - t1
        fpsOpencvDnn = frame_count / tt_opencvDnn

        for oldPoint in oldPoints:
            x,y = oldPoint.ravel()
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        label = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        frame = cv2.resize(frame, (480, 640))
        cv2.imshow("Sensomatic Miko", frame)

        if frame_count == 1:
            tt_opencvDnn = 0

        count += 1

        k = cv2.waitKey(1)
        if k == 27:
            break
   
    cap.release()
    cv2.destroyAllWindows()