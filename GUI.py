import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('handwritten.model')

# Set the title of the app
st.set_page_config(page_title="Handwritten Text Recognition", page_icon=":pencil2:")

# Add a title
st.title("Handwritten  digit Recognition")

# Add a file uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    # Preview the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    # Resize the image to 28x28
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert('L')
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Invert the image
    img_array = 255 - img_array
    # Normalize the image
    img_array = img_array / 255.0
    # Reshape the image to (1, 28, 28) as the model expects a batch of images
    img_array = img_array.reshape((1, 28, 28))
    # Predict the digit in the image
    digit = np.argmax(model.predict(img_array))
    # Show the predicted digit
    st.write("Predicted digit:", digit)
