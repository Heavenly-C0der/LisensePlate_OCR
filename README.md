
# 🚘 License Plate Detection and OCR App

A user-friendly web application built with Streamlit that detects vehicle license plates using a YOLOv8 object detection model and extracts text using Tesseract OCR.

---

## 🔍 Features

- 📤 Upload multiple vehicle images (`.jpg`, `.jpeg`, `.png`)
- 📦 License plate detection using YOLOv8 (`ultralytics`)
- 🔡 Plate number extraction using Tesseract OCR
- 🖼️ View annotated images with bounding boxes and recognized plate numbers
- 📥 Download annotated images and extracted plate numbers as text files

---

## 🧠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Detection Model:** [YOLOv8](https://docs.ultralytics.com/) via `ultralytics` Python package
- **OCR Engine:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- **Image Processing:** OpenCV, NumPy, PIL

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/license-plate-ocr-app.git
cd license-plate-ocr-app
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

- **Windows**: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  > Update the path in the script if necessary:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

- **macOS**:  
  ```bash
  brew install tesseract
  ```

- **Linux**:  
  ```bash
  sudo apt install tesseract-ocr
  ```

### 4. Add Your YOLOv8 Model

Place your trained license plate detection model at:

```
./models/license_plate_detector.pt
```

> You can train a custom YOLOv8 model using [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/).

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📁 Output Examples

- ✅ Annotated images with green bounding boxes and recognized plate text
- 📄 Text file with the extracted plate number(s)
- 🖼️ Visual confirmation and download options via the Streamlit interface

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit](https://streamlit.io/) for the easy-to-use interface

---
