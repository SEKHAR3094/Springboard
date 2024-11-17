import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os

# Path to the YOLOv5 repository and model
repo_path = '/mnt/src/springboard/yolov5'  # Adjust to your YOLOv5 repo location
model_path = '/mnt/src/springboard/Streamlit/best.pt'  # Adjust to the path where best.pt is located in the repo

# Load the YOLOv5 model
@st.cache_resource
def load_model(repo_path, model_path):
    return torch.hub.load(repo_path, 'custom', path=model_path, source='local')

model = load_model(repo_path, model_path)

# Main app interface
st.title("🎾 Tennis Tracking App")

# Upload video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open the video for reading
    cap = cv2.VideoCapture(temp_video_path)

    # Prepare the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        output_video_path = temp_output.name

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)
        processed_frame = np.squeeze(results.render())

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame
        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    # Release video resources
    cap.release()
    out.release()

    # Add download button
    with open(output_video_path, 'rb') as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_tennis_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
