import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pytesseract

# Set the full path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="True/False Evaluator", layout="wide")
st.title("✅ True/False Answer Evaluation System")

# === Load model ===
@st.cache_resource
def load_tf_model():
    return load_model(r"models\true_or_false_model.h5")

model = load_tf_model()
label_map = {0: "False", 1: "True"}

# === Function to extract ROIs ===
def extract_rois_from_image(image):
    all_rois = []

    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 80 <= w <= 180 and 30 <= h <= 60:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    for (x, y, w, h) in boxes:
        text_y_start = y + 30
        text_y_end = min(gray.shape[0], y + 35)
        roi = gray[y:text_y_end, x + 5:x + w - 5]
        if roi.shape[0] > 10 and roi.shape[1] > 20:
            roi_resized = cv2.resize(roi, (128, 32))
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            roi_final = np.expand_dims(roi_normalized, axis=-1)
            all_rois.append(roi_final)

    return all_rois

# === OCR using pytesseract ===
def perform_ocr_on_rois(rois):
    ocr_texts = []
    config = "--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for roi in rois:
        roi_uint8 = (roi.squeeze() * 255).astype(np.uint8)
        pil_img = Image.fromarray(roi_uint8)
        text = pytesseract.image_to_string(pil_img, config=config)
        ocr_texts.append(text)
    return ocr_texts

# === Standardize OCR output ===
def standardize_ocr_texts(ocr_texts):
    standardized = []
    for text in ocr_texts:
        cleaned = text.strip().lower()
        if "true" in cleaned:
            standardized.append("True")
        elif "false" in cleaned:
            standardized.append("False")
        else:
            standardized.append("Unknown")
    return standardized

# === File upload ===
col1, col2 = st.columns(2)
with col1:
    answer_key_file = st.file_uploader("📄 Upload Answer Key Image", type=["jpg", "png", "jpeg"], key="key")
with col2:
    answer_script_file = st.file_uploader("✍️ Upload Answer Script Image", type=["jpg", "png", "jpeg"], key="script")

if answer_key_file and answer_script_file:
    answer_key_img = Image.open(answer_key_file).convert("RGB")
    answer_script_img = Image.open(answer_script_file).convert("RGB")

    with st.spinner("🔍 Processing images..."):
        # === Extract ROIs ===
        answer_key_rois = extract_rois_from_image(answer_key_img)
        answer_script_rois = extract_rois_from_image(answer_script_img)

        # === OCR on answer_key ===
        ocr_texts = perform_ocr_on_rois(answer_key_rois)
        standardized_ocr = standardize_ocr_texts(ocr_texts)

        # === Model prediction on answer_script ===
        if len(answer_script_rois) == 0:
            st.error("❌ No valid ROIs found in answer script.")
        else:
            input_data = np.array(answer_script_rois)
            predictions = model.predict(input_data)
            predicted_labels = (predictions >= 0.5).astype(int).flatten()

            # === Scoring ===
            total_marks = 0
            marks_per_question = []
            for i in range(min(len(standardized_ocr), len(predicted_labels))):
                ocr_answer = standardized_ocr[i]
                predicted_answer = label_map[predicted_labels[i]]
                if ocr_answer == predicted_answer:
                    marks_per_question.append(1)
                    total_marks += 1
                else:
                    marks_per_question.append(0)

            # === Display Results ===
            st.success(f"🎯 Final Score: {total_marks} / {len(marks_per_question)}")

            # === Side-by-side view ===
            st.subheader("📊 Question-wise Comparison")

            for i in range(len(marks_per_question)):
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    st.image((answer_key_rois[i].squeeze() * 255).astype(np.uint8), caption=f"OCR: {standardized_ocr[i]}", use_container_width=True, clamp=True)
                with cols[1]:
                    st.image((answer_script_rois[i].squeeze() * 255).astype(np.uint8), caption=f"Pred: {label_map[predicted_labels[i]]}", use_container_width=True, clamp=True)
                with cols[2]:
                    mark = marks_per_question[i]
                    st.markdown(f"<h4 style='text-align: center;'>✅ Mark: {mark}</h4>", unsafe_allow_html=True)
else:
    st.info("👆 Please upload both images to continue.")
