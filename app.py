import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load saved model
model = load_model("cnn_image_classifier.h5")

# Class names for CIFAR-10
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Prediction function
def predict_image(img: Image.Image):
    img = img.resize((32, 32))  # Resize for the model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    return predicted_class, confidence

# Streamlit UI
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict the class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(img)
    st.markdown(f"**Prediction:** **{label}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
