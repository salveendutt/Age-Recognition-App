import cv2
import imutils
# Press ESC key to exit the while loop during runtime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('C:\\Users\\Hp\\Desktop\\Age recognition\\cascade.xml')
while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.17, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    # faces2 = face_cascade.detectMultiScale(
    #     imutils.rotate(gray_frame,angle=5), scaleFactor=1.17, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    
    # font = cv2.FONT_HERSHEY_SIMPLEX

    for(x, y, w, h) in faces:
        
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()



















