import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import tempfile
import platform
from ultralytics import YOLO

# Set Tesseract path if needed (especially on Windows)
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_resource
def load_model():
    return YOLO("./models/license_plate_detector.pt")

model = load_model()

st.title("License Plate Detection and OCR")

uploaded_files = st.file_uploader("Upload image(s) of vehicle(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model(image_rgb)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        annotated_image = image_rgb.copy()
        plate_number = ""
        plate_crop = None

        for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            plate_crop = annotated_image[y1:y2, x1:x2]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if plate_crop is not None:
                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate_number = pytesseract.image_to_string(thresh, config='--psm 7').strip()
                cv2.putText(annotated_image, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        st.image(annotated_image, caption=f"Detected Plate: {plate_number if plate_number else 'N/A'}", use_column_width=True)

        # Download buttons
        result_pil = Image.fromarray(annotated_image)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            result_pil.save(tmp.name)
            st.download_button("Download Annotated Image", data=open(tmp.name, "rb").read(), file_name=f"annotated_{uploaded_file.name}", mime="image/png")

        if plate_number:
            st.write("**Detected Plate Number:**", plate_number)
            st.download_button("Download Plate Text", plate_number, file_name=f"plate_{uploaded_file.name}.txt")
