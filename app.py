
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Road Accident Detection",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# LOAD YOLO MODEL FROM HUGGING FACE
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="Chithanoorimanoj/road-accident-detection",
        filename="best.pt"
    )

    model = YOLO(model_path)

    return model


try:
    with st.spinner("Loading AI accident detection model..."):
        model = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("🚗 AI Road Accident Detection & Severity Analysis")

st.markdown(
    "Upload a road image or video to detect accidents "
    "and analyze their severity using AI."
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

option = st.sidebar.selectbox(
    "Choose an option",
    ["Home", "Image Detection", "Video Detection", "About"]
)


# --------------------------------------------------
# SEVERITY CLASSIFICATION
# --------------------------------------------------

def get_severity(count):

    if count == 0:
        return "No Accident", "No accident detected"

    elif count <= 2:
        return "Low Severity", "Minor accident detected"

    elif count <= 5:
        return "Medium Severity", "Moderate accident detected"

    else:
        return "High Severity", "Multiple accident detections"


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if option == "Home":

    st.subheader("Welcome to AI Accident Detection System")

    st.write("""
    This application uses Artificial Intelligence and
    Deep Learning to detect road accidents and estimate
    accident severity.

    **Features:**

    - Road accident detection
    - Accident image analysis
    - Accident video analysis
    - Severity classification
    - Visual detection results
    """)

    st.success("AI Accident Detection Model Loaded Successfully!")


# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------

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

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        if st.button("Detect Accident"):

            with st.spinner("Analyzing image..."):

                results = model.predict(
                    source=image_array,
                    conf=0.25,
                    verbose=False
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


# --------------------------------------------------
# VIDEO DETECTION
# --------------------------------------------------

elif option == "Video Detection":

    st.subheader("Upload Road Accident Video")

    video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if video:

        if st.button("Analyze Video"):

            suffix = Path(video.name).suffix or ".mp4"

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            temp_path = temp_file.name

            try:

                temp_file.write(video.getbuffer())
                temp_file.close()

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error("Unable to open the uploaded video.")
                    st.stop()

                stframe = st.empty()

                frame_count = 0
                accident_frames = 0

                with st.spinner("Analyzing video..."):

                    while cap.isOpened():

                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame_count += 1

                        # Analyze every fifth frame

                        if frame_count % 5 == 0:

                            results = model.predict(
                                source=frame,
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

                severity, description = get_severity(
                    accident_frames
                )

                st.subheader("Video Analysis Results")

                st.metric("Analyzed Frames", frame_count)

                st.metric("Detection Frames", accident_frames)

                st.success(
                    f"Estimated Severity: {severity}"
                )

                st.write(description)

            except Exception as e:

                st.error(f"Video processing error: {e}")

            finally:

                if 'cap' in locals():
                    cap.release()

                if os.path.exists(temp_path):
                    os.remove(temp_path)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif option == "About":

    st.subheader("About This Project")

    st.write("""
    **Project Title:** AI Road Accident Detection and
    Severity Analysis

    **Technologies Used:**

    - Python
    - Streamlit
    - YOLO Object Detection
    - OpenCV
    - NumPy
    - Hugging Face Model Hosting

    **Project Objective:**

    To detect road accidents from uploaded images and
    videos using deep learning and provide an estimated
    severity classification based on detection results.
    """)

    st.info(
        "Developed as an AI-based Road Accident Detection "
        "and Severity Analysis project."
    )



