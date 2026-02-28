import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ----------------------------
# Download model from Drive if not present
# ----------------------------
model_path = "healthy_vs_rotten.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1aV9iZgFbmJ6-4gYL8Ek_EpX41Ozct_76"
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Class labels (same order as training)
class_names = [
    'Apple_Healthy', 'Apple_Rotten',
    'Banana_Healthy', 'Banana_Rotten',
    'Bellpepper_Healthy', 'Bellpepper_Rotten',
    'Carrot_Healthy', 'Carrot_Rotten',
    'Cucumber_Healthy', 'Cucumber_Rotten',
    'Grape_Healthy', 'Grape_Rotten',
    'Guava_Healthy', 'Guava_Rotten',
    'Jujube_Healthy', 'Jujube_Rotten',
    'Mango_Healthy', 'Mango_Rotten',
    'Orange_Healthy', 'Orange_Rotten',
    'Pomegranate_Healthy', 'Pomegranate_Rotten',
    'Potato_Healthy', 'Potato_Rotten',
    'Strawberry_Healthy', 'Strawberry_Rotten',
    'Tomato_Healthy', 'Tomato_Rotten'
]

st.title("🍎 Smart Sorting - Fruit & Vegetable Freshness Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
