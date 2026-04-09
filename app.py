import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(page_title="Food Classifier", layout="centered")

# Load model
@st.cache_resource
def load_my_model():
    return load_model("street_food_model.h5", Compile=False)

model = load_my_model()

# Class labels (make sure order matches your training)
class_names = ['Vada Pav', 'Paani Puri', 'Masala Dosa', 'Idli', 'Samosa', 'Grilled Sandwich']

# Title
st.title("Mumbai Street Food Classifier")
st.write("Upload an image and the model will predict the food category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")
