from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot 
from numpy import asarray

import cv2

# model = MTCNN(weights_file='weights.npy', min_face_size=30, scale_factor=0.709)
detector = MTCNN(scale_factor=0.709)

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if ret:
        # scale_percent = 100 # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        pix_frame = asarray(frame)
        
        faces = detector.detect_faces(pix_frame)
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 2)
        
        cv2.imshow('frame', frame)
    else:
        print('No Camera found')
        cap.release()
        cv2.destroyAllWindows()
        break
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
    


