
import streamlit as st

import numpy as np
from pyngrok import ngrok
import tensorflow as tf

from src import inference
from src.training import training_utils

MODEL = tf.keras.models.load_model('data/simpleshopping-model')
LABELS = training_utils.read_files('data/labels.txt')

def prediction(image: np.array, 
               image_dims: tuple = (256, 256),
               labels: list = LABELS):
    image = tf.image.resize(image, image_dims)
    image = tf.image.convert_image_dtype(
        images=image, 
        dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    label_index, probs = inference.run_inference(image, MODEL)
    class_name = labels[label_index]
    return class_name, label_index, probs

def website():
    test = 'test string'
    st.set_page_config(layout="wide")

    original_title = '<p style="font-family:monospace; color:Black; font-size: 50px;">SimpleShopping</p>'
    st.markdown(original_title, unsafe_allow_html=True)



    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown("<h1 style='text-align: center; font-size: 30px; color: #a1ae25; padding-bottom: 1px'>Camera Stream</h1>", unsafe_allow_html=True)
        col1.camera_input("")


    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 30px; color: black;'>Product List</h1>", unsafe_allow_html=True)
        st.write(test)
        backgroundSettings = '<div color: #a1ae25; height: 100%; width: 100%;"></div>'
        st.markdown(backgroundSettings, unsafe_allow_html=True)
        
    with col3:
        st.markdown("<h1 style='text-align: center; font-size: 30px; color: #a1ae25;'>Recipes</h1>", unsafe_allow_html=True)

    public_url = ngrok.connect(port='80')
    print (public_url)

if __name__ == '__main__':
    website()