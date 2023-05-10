from streamlit_option_menu import option_menu
import cvlib as cv
import cv2
import streamlit as st
import numpy as np
from keras.models import load_model

#-----------------------------------------------------------------------------------------------------------------------
# model = load_model("C:\\Users\\mrsal\\Github Repositories\\Age-Recognition-App\\app\\models\\CNN_MODEL_64.h5")
model = load_model("C:\\Users\\mrsal\\Github Repositories\\Age-Recognition-App\\app\\models\\CNN_MODEL_64_CHATGPT.h5")

def getAge(age):
    if age == 0:
        return "[1-9]"
    if age == 1:
        return "[10-15]"
    if age == 2:
        return "[16-20]"
    if age == 3:
        return "[21-27]"
    if age == 4:
        return "[28-34]"
    if age == 5:
        return "[35-46]"
    if age == 6:
        return "[47-65]"
    if age == 7:
        return "[65-100]"

# Function to capture video frames
def capture_video():
    cap = cv2.VideoCapture(0) # Open the default camera (index 0)
    while True:
        ret, frame = cap.read() # Read a frame from the camera
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB format
        yield frame
        
        
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv2.equalizeHist(img)
    img = img / 255  # normalizing image.
    img = cv2.resize(img, (64, 64))  # resizing it.
    return img


def predict(image):
    img = [image]
    processed_img = []
    for x in img:
        processed_img.append(preprocessing(x))
    processed_img = np.array(processed_img)
    processed_img = processed_img.reshape(processed_img.shape[0], processed_img.shape[1], processed_img.shape[2], 1)
    result = model.predict(processed_img)
    predictedClass = np.argmax(result)
    return predictedClass
#-----------------------------------------------------------------------------------------------------------------------
# Define Streamlit app pages
def home_page():
    st.title("The Project of Face Recognition Application")
    st.write("Welcome to our home page!")
    st.markdown("***Authors***: Salveen Dutt, Kamilla Tadjibaeva, \
        Seita Fujiwara, Cansu Ustunel")

def camera_page():
    st.title("Face Recognition App - Use Camera")
    st.write("This is the camera page.")
    
    # Create a canvas to display video frames
    video_canvas = st.empty()
    
    # Start capturing video frames
    video_generator = capture_video()
    for frame in video_generator:
        faces, _ = cv.detect_face(frame) # Detect faces using cvlib
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        video_canvas.image(frame, channels="RGB", use_column_width=True)
    # Release the webcam when app is closed
    del video_generator
    cv2.destroyAllWindows()

def picture_page():
    st.title("Face Recognition App - Use Picture")
    st.write("This is the picture page.")

    # Add code for using a picture here
    ages = []
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces using cvlib
        faces, _ = cv.detect_face(image)
        for face in faces:
            x1, y1, x2, y2 = face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detectedFace = image[y1:y2, x1:x2]
            ageClass = predict(detectedFace)
            ages.append(ageClass)

        for i, ageClass in enumerate(ages):
            age = getAge(ageClass)
            text_size, baseline = cv2.getTextSize(age, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
            text_width, text_height = text_size
            x = faces[i][0] + (faces[i][2] - faces[i][0]) // 2 - text_width // 2
            y = faces[i][1] - text_height // 2
            cv2.putText(image, age, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

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
