import cvlib as cv
import cv2

def readImage(image_path):
    img = cv2.imread(image_path)
    faces, _ = cv.detect_face(img)
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img,(x,y), (w, h), (0,255,0), 2)
    print('Found {0} faces'.format(len(faces)))
    print(faces)
    return len(faces)

for i in range(1,5):
    readImage('C:\\Users\\Hp\\Pictures\\Camera Roll\\{0}.jpg'.format(i))
    