import base64
import glob
import cv2
import numpy as np
import streamlit as st
import detect_age as da
from streamlit_option_menu import option_menu
import time
import multiprocessing as mp

captured_pictures = []

def take_picture(video_generator):

    frame = next(video_generator)
    success_img = st.image(frame, channels="RGB", use_column_width=True)
    timestamp = int(time.time())
    name = f"captured_picture_{timestamp}.jpg"

    image, faces = da.detect_faces(frame)
    predictions = da.make_predictions(image, faces)
    for face, prediction in zip(faces, predictions):
        age = prediction['age']
        emotion = prediction['emotion']
        da.draw_age_label(image, face, age, emotion)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR color space
    cv2.imwrite(name, image_bgr)

    captured_pictures = st.session_state.get("captured_pictures", [])
    captured_pictures.append(name)
    st.session_state["captured_pictures"] = captured_pictures

    success_message = st.success("Picture captured and saved!")
    # delete after 1 second
    time.sleep(1)
    success_img.empty()
    success_message.empty()


def capture_video():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

# Define Streamlit app pages
def home_page():
    st.title("The Project of Face Recognition Application")
    st.write("Welcome to our home page!")
    st.markdown("***Authors***: Salveen Dutt, Kamilla Tadjibaeva, \
        Seita Fujiwara, Cansu Ustunel")


# Function for the picture page
def picture_page():
    st.title("Face Recognition App - Use Picture")
    st.write("Please load your picture")
    # Add code for using a picture here
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = da.detect_faces_and_predict(image)
        image, faces = da.detect_faces(image)
        predictions = da.make_predictions(image, faces)
        for face, prediction in zip(faces, predictions):
            age = prediction['age']
            emotion = prediction['emotion']
            da.draw_age_label(image, face, age, emotion)
        
        st.image(image, channels="RGB", use_column_width=True)


from threading import Thread
import time
def age_prediction_thread(image, face, predictions, index):
    detected_face = da.extract_detected_face(image, face)
    predicted_age = da.predict_age(detected_face)
    predictions[index]['age'] = predicted_age

def emotion_prediction_thread(image, face, predictions, index):
    detected_face = da.extract_detected_face(image, face)
    predicted_emotion = da.predict_emotion(detected_face)
    predictions[index]['emotion'] = predicted_emotion

def camera_page():
    st.title("Face Recognition App - Use Camera")
    st.write("This is the camera page.")
    video_canvas = st.empty()
    video_generator = capture_video()
    predictions = []
    threads = []  # Initialize the threads list

    if st.button("Take a Picture"):
        take_picture(video_generator)
    
    start_time = time.time()
    for frame in video_generator:

        image, faces = da.detect_faces(frame)
        
        # Create new predictions if number of faces changed
        if len(predictions) != len(faces):
            predictions = [{'age': 'Unknown', 'emotion': 'Unknown'} for _ in range(len(faces))]
        
        # Perform threading for age and emotion prediction every 1.5 seconds
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1:
            threads = []
            for i, (face, prediction) in enumerate(zip(faces, predictions)):
                
                age_thread = Thread(target=age_prediction_thread, args=(image, face, predictions, i))
                age_thread.start()
                threads.append(age_thread)
                
                emotion_thread = Thread(target=emotion_prediction_thread, args=(image, face, predictions, i))
                emotion_thread.start()
                threads.append(emotion_thread)
            
            start_time = current_time
        
        for face, prediction in zip(faces, predictions):
            age = prediction['age']
            emotion = prediction['emotion']
            da.draw_age_label(image, face, age, emotion)
        
        video_canvas.image(image, channels="RGB", use_column_width=True)
    
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    
    del video_generator
    cv2.destroyAllWindows()

def gallery_page():
    st.title("Face Recognition App - Picture Gallery")
    st.write("This is the picture gallery page.")

    picture_files = st.session_state.get("captured_pictures", [])

    if len(picture_files) == 0:
      st.write("No pictures captured yet.")
    else:
        # Display each captured picture in a grid layout
        columns = 4
        for i in range(0, len(picture_files), columns):
            pictures_row = picture_files[i:i+columns]
            col_list = st.columns(columns)

            for col, picture_file in zip(col_list, pictures_row):
                image = cv2.imread(picture_file)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                col.image(image_rgb, channels="RGB", use_column_width=True)

                # Add a download button for each picture
                download_button_caption = "Download"
                if col.button(download_button_caption, key=picture_file):
                    with open(picture_file, "rb") as file:
                        file_data = file.read()
                        encoded_file = base64.b64encode(file_data).decode("utf-8")
                        href = f'data:application/octet-stream;base64,{encoded_file}'
                        download_link = f'<a href="{href}" download="{picture_file}">Click here to download</a>'
                        col.markdown(download_link, unsafe_allow_html=True)

# Main Streamlit app
def app():
    selected = option_menu(
        menu_title="Face Recognition App",
        options=["Home", "Open Camera", "Open Picture", "Gallery"],
        icons=["house", "camera-fill", "image", "collection"],
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Home":
        home_page()
    elif selected == "Open Camera":
        camera_page()
    elif selected == "Open Picture":
        picture_page()
    elif selected == "Gallery":
        gallery_page()

if __name__ == "__main__":
    app()