import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page layout
st.set_page_config(layout="centered")

# Load model
@st.cache_resource
def load_model_file():
    return tf.keras.models.load_model("food_model.keras")

model = load_model_file()

# Class labels
class_names = ["Grilled Sandwich", "Idli", "Masala Dosa", "Paani Puri", "Samosa", "Vada Pav"]

# Title
st.title("🍽️ Mumbai Street Food Classification App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).resize((224, 224))

    # Create nice layout
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(img, caption="Uploaded Image", width=220)

    # Preprocess
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    with col2:
        st.markdown("### Prediction Result")
        st.success(f"🍽️ {class_names[predicted_class]}")
        st.info(f"📊 Confidence: {confidence * 100:.2f}%")

