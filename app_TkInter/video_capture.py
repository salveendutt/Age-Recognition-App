import cvlib as cv
import cv2

def start_video():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if(ret):
            cv2.resize(frame,(1280,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            faces, _ = cv.detect_face(frame)
            
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(frame,(x,y), (w, h), (0,255,0), 2)
        
            cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
