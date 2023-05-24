import base64
import glob
import cv2
import numpy as np
import streamlit as st
import detect_age as da
from streamlit_option_menu import option_menu
import multiprocessing as mp

INDX = 1;

def take_picture(video_generator):
    global INDX
    frame = next(video_generator)
    st.image(frame, channels="RGB", use_column_width=True)
    name = "captured_picture" + str(INDX) + ".jpg"
    cv2.imwrite(name, frame)
    INDX = INDX + 1
    st.success("Picture captured and saved!")


def capture_video():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame


def camera_page():
    st.title("Face Recognition App - Use Camera")
    st.write("This is the camera page.")
    video_canvas = st.empty()
    video_generator = capture_video()
    if st.button("Take a Picture"):
        take_picture(video_generator)
    for frame in video_generator:
        image = da.detect_faces_and_predict_age(frame)
        video_canvas.image(image, channels="RGB", use_column_width=True)
    del video_generator
    cv2.destroyAllWindows()

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
        image = da.detect_faces_and_predict_age(image)
        # Display the image with predicted ages
        st.image(image, channels="RGB", use_column_width=True)

def gallery_page():
    st.title("Face Recognition App - Picture Gallery")
    st.write("This is the picture gallery page.")
    # Get the list of captured pictures
    picture_files = sorted(glob.glob("captured_picture*.jpg"))
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
                col.image(image_rgb, channels="RGB", use_column_width=True, caption=picture_file)
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