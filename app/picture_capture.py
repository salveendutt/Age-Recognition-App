import cvlib as cv
import cv2
import numpy as np
import video_capture as vc

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv2.equalizeHist(img)
    img = img / 255  # normalizing image.
    img = cv2.resize(img, (64, 64))  # resizing it.
    return img

def predict(image):
    img = []
    img.append(image)
    processedImg = []
    for x in img:
        processedImg.append(preprocessing(x))
    processedImg = np.array(processedImg)
    processedImg = processedImg.reshape(processedImg.shape[0], processedImg.shape[1], processedImg.shape[2], 1)
    result = vc.model.predict(processedImg)
    predictedClass = np.argmax(result)
    return predictedClass

def readImage(image_path):
    ages = []
    img = cv2.imread(image_path)
    faces, _ = cv.detect_face(img)
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        detectedFace = img[y:y + h, x:x + w]
        ageClass = predict(detectedFace)
        ages.append(ageClass)
    print('Found {0} faces'.format(len(faces)))
    cv2.resize(img, (300, 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    thickness = 2
    for i, ageClass in enumerate(ages):
        age = vc.getAge(ageClass)
        text_size, baseline = cv2.getTextSize(age, font, fontScale=0.5, thickness=2)
        text_width, text_height = text_size
        x = faces[i][0] + (faces[i][2] - text_width) // 2
        y = faces[i][1] - text_height
        cv2.putText(img, age, (x, y), font, fontScale=1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    return len(faces)
