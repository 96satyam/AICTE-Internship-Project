import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    # Loading the model
    model = tf.keras.models.load_image('plant_desease_trained_model.keras')
    
    # Loading the image
    img = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])  

    # Making a prediction
    predictions = model.predict(img_array)
    #score = tf.nn.softmax(predictions[0])

    return np.argmax(predictions)

st.sidebar.title("Potato plant Desease system for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('select page', ["Home","Desease Recoginization"])

from PIL import Image
img = Image.open("Diseases.png")
st.image(img)

if(app_mode == "Home"):
    st.markdown("<h1 style = 'test-align:center;'>Welcome to Potato plant Desease system for Sustainable Agriculture", unsafe_allow_html=True)
   