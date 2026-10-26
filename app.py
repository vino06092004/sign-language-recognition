import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="ASL Sign Recognition",
    page_icon="🤟",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "sign_language_model.keras")
CLASSES_PATH = os.path.join("models", "classes.txt")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

model, classes = load_trained_model()

st.title("🤟 American Sign Language (ASL) Recognition")
st.write("Upload a photo of an ASL alphabet sign or take a snapshot with your webcam to identify the gesture.")

if model is None:
    st.error("⚠️ Model files not found in the `models/` folder.")
    st.info("Run `python train.py` in your VS Code terminal first to train the network and save the model.")
    st.stop()

# Mode selection
mode = st.radio("Choose Input Mode:", ["Upload Image", "Webcam Snapshot"], horizontal=True)
image_input = None

if mode == "Upload Image":
    file = st.file_uploader("Upload hand sign image:", type=["jpg", "jpeg", "png"])
    if file:
        image_input = Image.open(file)
else:
    file = st.camera_input("Take a snapshot of your hand gesture:")
    if file:
        image_input = Image.open(file)

# Display & Prediction
if image_input:
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Gesture")
        st.image(image_input, caption="Captured Image", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        with st.spinner("Analyzing gesture..."):
            # Resize directly to match training input (64x64)
            img = image_input.convert("RGB").resize((64, 64))
            img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

            # Predict
            preds = model.predict(img_arr)[0]
            best_idx = int(np.argmax(preds))
            confidence = float(preds[best_idx]) * 100
            label = classes[best_idx]

        st.metric(label="Detected Sign", value=label)
        st.progress(float(confidence / 100))
        st.write(f"Confidence: **{confidence:.1f}%**")

        if confidence < 50.0:
            st.warning("⚠️ Low confidence. Make sure your hand is centered and the lighting is clear.")

        st.write("---")
        st.write("**Top 3 Predictions:**")
        top_3 = np.argsort(preds)[-3:][::-1]
        for idx in top_3:
            st.write(f"- **{classes[idx]}**: `{preds[idx] * 100:.1f}%`")