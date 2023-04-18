from streamlit_option_menu import option_menu
import cv2
import streamlit as st
import numpy as np


# Function to capture video frames
def capture_video():
    cap = cv2.VideoCapture(0) # Open the default camera (index 0)
    while True:
        ret, frame = cap.read() # Read a frame from the camera
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB format
        yield frame

# Define Streamlit app pages
def home_page():
    st.title("Face Recognition App - Home")
    st.write("Welcome to the home page!")

def camera_page():
    st.title("Face Recognition App - Use Camera")
    st.write("This is the camera page.")
    
    # Create a canvas to display video frames
    video_canvas = st.empty()
    
    # Start capturing video frames
    video_generator = capture_video()
    for frame in video_generator:
        video_canvas.image(frame, channels="RGB", use_column_width=True)
        
    # Release the webcam when app is closed
    del video_generator
    cv2.destroyAllWindows()

def picture_page():
    st.title("Face Recognition App - Use Picture")
    st.write("This is the picture page.")
    
    # Add code for using a picture here
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, channels="RGB", use_column_width=True)

# Main Streamlit app
def app():
    selected = option_menu(
        menu_title="Face Recognition App",
        options=["Home", "Use Camera", "Use Picture"],
        icons=["house", "camera-fill", "image"],
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Home":
        home_page()
    elif selected == "Use Camera":
        camera_page()
    elif selected == "Use Picture":
        picture_page()

if __name__ == "__main__":
    app()
