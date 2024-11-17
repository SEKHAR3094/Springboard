import sys
import os

# Add yolov5 directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

import torch
import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import time

# Paths
repo_path = './yolov5'  # Path to the yolov5 folder
model_path = './yolov5/best.pt'  # Path to the best.pt model inside yolov5 folder

# Load the YOLOv5 model using YOLOv5's custom load method
from yolov5 import load  # Import load function from yolov5
model = load(model_path)  # Load model with YOLOv5's method

# Streamlit app UI
st.title('Tennis Player Detection App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = temp_video.name
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)  # Use model for detection
        frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

        # Limit frame rate
        time.sleep(0.03)

    cap.release()
    st.success('Video processing complete!')

    # Cleanup
    os.unlink(temp_video_path)

st.write("Ensure 'best.pt' is in the yolov5 directory or provide the correct path in model_path.")
