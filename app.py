import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# 🔁 Télécharger le modèle depuis Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1FvddbOX0gGcZV7BPxLPjZsiNQi9o3H9d"
MODEL_PATH = "pneumonia_detector_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔁 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Charger le modèle
model = load_model(MODEL_PATH)

st.title("🩺 Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0])

    if prediction > 0.5:
        st.error(f"🟥 Pneumonia detected with confidence {prediction:.2f}")
    else:
        st.success(f"🟩 Normal with confidence {1 - prediction:.2f}")

