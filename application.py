import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_desease_trained_model.keras')

model = load_model()

def preprocess_image(image):
    img = Image.open(image).resize((128, 128))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def model_prediction(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return np.argmax(predictions)  


st.sidebar.title("Potato Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ["Home", "Disease Recognition"])


img = Image.open("Diseases.png")
st.image(img)

if app_mode == "Home":
    st.markdown("<h1 style='text-align:center;'>Welcome to the Potato Plant Disease System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("Potato Plant Disease Recognition System for Sustainable Agriculture")

    test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)  
        
        if st.button('Predict'):
            st.snow()
            result_index = model_prediction(test_image)

            class_names = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']
            st.success(f"Model Prediction: **{class_names[result_index]}**")
