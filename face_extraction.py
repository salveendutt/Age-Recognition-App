import cvlib as cv
import cv2
def get_face():
    faces, _ = cv.detect_face(frame)
    
    for face in faces:
        x, y, w, h = face
        # cv2.rectangle(frame,(x,y), (w, h), (0,255,0), 2)
        cv2.imwrite(str(age + "_" + str(count)) + '.jpg', face)
