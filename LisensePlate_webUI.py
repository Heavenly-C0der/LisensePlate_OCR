import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="License Plate OCR", layout="centered")

st.title("🚗 License Plate Detection & OCR")

# Load YOLOv5 model from Ultralytics
from ultralytics import YOLO
model = YOLO("yolov5s.pt")  # or your own plate detection model

reader = easyocr.Reader(['en'])

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Original Image", use_column_width=True)

    st.subheader("🔍 Detecting license plate...")
    results = model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    if len(boxes) == 0:
        st.warning("No license plate detected.")
    else:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]
            st.image(plate_img, caption=f"Detected Plate #{i+1}", use_column_width=False)

            st.subheader("📝 OCR Result")
            plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            result = reader.readtext(plate_rgb)

            if result:
                for detection in result:
                    text = detection[1]
                    conf = detection[2]
                    st.success(f"Detected Text: `{text}` (Confidence: {conf:.2f})")
            else:
                st.warning("No text detected by OCR.")
