import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Chithanoorimanoj/road-accident-detection",
        filename="best.pt",
        token=st.secrets.get("HF_TOKEN")
    )
    return YOLO(model_path)

model = load_model()

# Page configuration
st.set_page_config(
    page_title="AI Road Accident Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom title
st.title("🚗 AI Road Accident Detection & Severity Analysis")
st.markdown(
    "Upload a road image or video to detect accidents "
    "and analyze their severity using AI."
)

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option",
    ["Home", "Image Detection", "Video Detection", "About"]
)

# Severity classification
def get_severity(count):
    if count == 0:
        return "No Accident", "No accident detected"
    elif count <= 2:
        return "Low Severity", "Minor accident detected"
    elif count <= 5:
        return "Medium Severity", "Moderate accident detected"
    else:
        return "High Severity", "Multiple accident detections"

# Home page
if option == "Home":
    st.subheader("Welcome to AI Accident Detection System")

    st.write("""
    This application uses Artificial Intelligence and Deep Learning
    to detect road accidents and estimate accident severity.

    **Features:**
    - Road accident detection
    - Accident image analysis
    - Accident video analysis
    - Severity classification
    - Visual detection results
    """)

# Image detection
elif option == "Image Detection":

    st.subheader("Upload Accident Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image",
                     use_container_width=True)

        if st.button("Detect Accident"):

            with st.spinner("Analyzing image..."):

                results = model.predict(
                    source=image_array,
                    conf=0.25
                )

                result_image = results[0].plot()
                result_image = cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                )

                boxes = results[0].boxes
                count = len(boxes) if boxes is not None else 0

                severity, description = get_severity(count)

            with col2:
                st.image(
                    result_image,
                    caption="Detection Result",
                    use_container_width=True
                )

            st.subheader("Accident Analysis")

            st.metric("Detected Objects", count)

            st.success(f"Severity: {severity}")
            st.write(description)

# Video detection
elif option == "Video Detection":

    st.subheader("Upload Road Accident Video")

    video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if video:
        if st.button("Analyze Video"):

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_file.write(video.read())
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)

            stframe = st.empty()

            frame_count = 0
            accident_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if frame_count % 5 == 0:

                    results = model.predict(
                        frame,
                        conf=0.25,
                        verbose=False
                    )

                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        accident_frames += 1

                    output = results[0].plot()
                    output = cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB
                    )

                    stframe.image(
                        output,
                        channels="RGB",
                        use_container_width=True
                    )

            cap.release()
            os.unlink(temp_file.name)

            severity, description = get_severity(
                accident_frames
            )

            st.subheader("Video Analysis Results")
            st.metric("Analyzed Frames", frame_count)
            st.metric("Detection Frames", accident_frames)
            st.success(f"Estimated Severity: {severity}")
            st.write(description)

# About page
elif option == "About":

    st.subheader("About This Project")

    st.write("""
    Project Title: AI Road Accident Detection and Severity Analysis

    Technologies Used:
    - Python
    - YOLO Object Detection
    - OpenCV
    - Deep Learning
    - Streamlit

    Objective:
    To detect road accidents automatically and estimate
    their severity using computer vision techniques.
    """)
