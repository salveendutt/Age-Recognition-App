import cv2
# import imutils
# Press ESC key to exit the while loop during runtime
import os

cap = cv2.VideoCapture(0)


# while True:
ret, frame = cap.read()
count = 0


def getFace(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # img = cv2.imread("img.jpg")
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4)
    allFaces = []
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]  # slice the face from the image
        allFaces.append(face)
    return allFaces
















