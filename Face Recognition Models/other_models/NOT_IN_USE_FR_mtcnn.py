from mtcnn.mtcnn import MTCNN
from numpy import asarray

import cv2
# Press ESC key to exit the while loop during runtime

detector = MTCNN(scale_factor=0.709)

cap = cv2.VideoCapture(0)
while True:
    
    ret, frame = cap.read()
    if ret:
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
    


