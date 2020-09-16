import cv2
from darkflow.net.build import TFNet
import numpy as np
import time





option = {
#old
   'model': 'cfg/yolo.cfg',
    
   'load': 'bin/yolo.weights',  

#new    
#     'model': 'cfg/yolov2-tiny.cfg',
#     'load': 1875,
    'threshold': 0.5,
    'gpu': 0.8
}
tfnet = TFNet(option)
capture = cv2.VideoCapture('test2.mp4')
#capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            print(label)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break