import cvlib as cv
import cv2
import streamlit as st
from keras.models import load_model

model = load_model("./app/models/resnet.h5")


def preprocessing(given_img):
    if given_img.size == 0:
        raise ValueError("Input image is empty.")
    given_img = given_img / 255  # normalizing image.
    given_img = cv2.resize(given_img, (64, 64))  # resizing it.
    return given_img

def predict_age(image):
    processed_img = preprocessing(image)
    processed_img = processed_img.reshape(1, 64, 64, 3)
    predicted_age = model.predict(processed_img)[0]
    return int(predicted_age)

def draw_face_rectangles(image, faces):
    for face in faces:
        x1, y1, x2, y2 = face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

def extract_detected_face(image, face):
    x1, y1, x2, y2 = face
    detected_face = image[y1:y2, x1:x2]
    return detected_face

def draw_age_label(image, face, predicted_age):
    x1, y1, x2, _ = face
    x = x1 + (x2 - x1) // 2
    y = y1 - 10
    cv2.putText(image, str(predicted_age), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(0, 255, 0), thickness=2)

def detect_faces_and_predict_age(image):
    faces, conf = cv.detect_face(image)
    if len(faces) == 0:
        return image  # Return the original image if no faces are detected
    draw_face_rectangles(image, faces)
    frame_height, frame_width, _ = image.shape
    for face in faces:
        x1, y1, x2, y2 = face
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            continue
        detected_face = extract_detected_face(image, face)
        predicted_age = predict_age(detected_face)
        draw_age_label(image, face, predicted_age)
    return image

def my_capture_video():
    cap = cv2.VideoCapture(0) 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()