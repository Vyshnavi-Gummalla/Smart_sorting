import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("healthy_vs_rotten.h5")

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

    # Resize to match training size
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Normalize (IMPORTANT if used during training)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
