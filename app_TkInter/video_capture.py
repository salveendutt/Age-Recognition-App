import cvlib as cv
import cv2
from keras.models import load_model
import numpy as np
import os
import pictureCaptureV2 as pc
model = load_model("CNN_MODEL_12EPCOCHS.h5")


def getAge(age):
    if age == 0:
        return "Age: [1-2]"
    if age == 1:
        return "Age: [3-9]"
    if age == 2:
        return "Age: [10-20]"
    if age == 3:
        return "Age: [21-27]"
    if age == 4:
        return "Age: [28-34]"
    if age == 5:
        return "Age: [35-46]"
    if age == 6:
        return "Age: [47-65]"
    if age == 7:
        return "Age: [65-100]"


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv2.equalizeHist(img)
    img = img/255  # normalizing image.
    img = cv2.resize(img, (64, 64))  # resizing it.
    return img

def predict(image):
    img = []
    img.append(image)
    processedImg= []
    for x in img:
        processedImg.append(preprocessing(x))
    processedImg = np.array(processedImg)
    processedImg = processedImg.reshape(processedImg.shape[0], processedImg.shape[1], processedImg.shape[2], 1)
    result = model.predict(processedImg)
    predictedClass = np.argmax(result)
    #print(predictedClass)
    return predictedClass

def start_video():
    cap = cv2.VideoCapture(0)
    ages = []
    currentAgeString = ""
    while True:
        ret, frame = cap.read()

        if(ret):
            cv2.resize(frame,(1280,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            faces, _ = cv.detect_face(frame)
            
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(frame,(x,y), (w, h), (0,255,0), 2)
                detectedFace = frame[y:y + h, x:x + w]
                ageClass = predict(detectedFace)
                ages.append(ageClass)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (50, 50)
                # fontScale
                fontScale = 1
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2
                # Using cv2.putText() method
                i = 0
            for x in ages:
                if(i == 0):
                    image = cv2.putText(frame, getAge(x), org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                    i = i + 1
            cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
