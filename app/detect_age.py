import cvlib as cv
import cv2
import streamlit as st
from keras.models import load_model
import numpy as np

model_age = load_model("./app/models/resnet_trainable.h5")
model_emotion = load_model("./app/models/emotion.h5")

emotion_labels = ['Happy', 'Surprise', 'Sad', 'Neutral', 'Angry', 'Disgust', 'Fear']

def preprocess_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    preprocessed = normalized.reshape(1, 48, 48, 1)
    return preprocessed

def preprocess_age(given_img):
    if given_img.size == 0:
        raise ValueError("Input image is empty.")
    given_img = given_img / 255
    given_img = cv2.resize(given_img, (64, 64))  
    return given_img

def predict_age(image):
    processed_img = preprocess_age(image)
    processed_img = processed_img.reshape(1, 64, 64, 3)
    predicted_age = model_age.predict(processed_img)[0]
    return int(predicted_age)

def predict_emotion(image):
    preprocessed_image = preprocess_emotion(image)
    predictions = model_emotion.predict(preprocessed_image)
    predicted_label_index = np.argmax(predictions[0])
    predicted_emotion = emotion_labels[predicted_label_index]
    return predicted_emotion

def draw_face_rectangles(image, faces):
    for face in faces:
        x1, y1, x2, y2 = face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

def extract_detected_face(image, face):
    x1, y1, x2, y2 = face
    detected_face = image[y1:y2, x1:x2]
    return detected_face

def draw_age_label(image, face, predicted_age, predicted_emotion):
    x1, y1, x2, y2 = face
    x = x1 + (x2 - x1) // 2
    y = y1 - 10
    cv2.putText(image, str(predicted_age), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.putText(image, predicted_emotion, (x, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(0, 255, 0), thickness=2)

def detect_faces_and_predict(image):
    faces, conf = cv.detect_face(image)
    if len(faces) == 0:
        return image 
    draw_face_rectangles(image, faces)
    frame_height, frame_width, _ = image.shape
    for face in faces:
        x1, y1, x2, y2 = face
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            continue
        detected_face = extract_detected_face(image, face)
        predicted_age = predict_age(detected_face)
        predicted_emotion = predict_emotion(detected_face)
        draw_age_label(image, face, predicted_age, predicted_emotion)
    return image

def my_capture_video():
    cap = cv2.VideoCapture(0) 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()