import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("YOLOv8 Live Video Stream 🎥🚀")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or your custom best.pt

# Sidebar controls
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Start webcam stream
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Run YOLO inference
        results = model(frame, conf=confidence)

        # Draw detections
        annotated_frame = results[0].plot()

        # Streamlit displays images in RGB, so convert
        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    cap.release()
