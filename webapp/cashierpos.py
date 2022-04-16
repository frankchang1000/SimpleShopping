# SimplisticShopping 2022
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Thomas Chia (IdeaKing) and Frank Chang (FrankChang1000)
# Created Date: 16/04/2020 
# version = 1.0.0
# license: Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)
# ---------------------------------------------------------------------------
"""Runs Streamlit App."""
# ---------------------------------------------------------------------------

import streamlit as st

import numpy as np
from pyngrok import ngrok
import tensorflow as tf

from src import inference
from src.training import training_utils
from . import barcode, recipes_foods, file_reader


MODEL = tf.keras.models.load_model("data/simpleshopping-model")
print("Finished model loading...")
LABELS = training_utils.read_files("data/labels.txt")
print("Finished loading labels...")
NUTRITION = recipes_foods.NutritionInfo(
    consumer_key=file_reader.read_keys()["consumer_key"],
    consumer_secret=file_reader.read_keys()["consumer_secret"])
print("Finished nutrition info...")


def prediction(image: np.array,
               image_dims: tuple = (512, 512),
               labels: list = None):
    """Predicts the image.
    Params:
        image: np.array
            The image.
        image_dims: tuple
            The image dimensions.
        labels: list
            The labels.
    Returns:
        preds: list
            The predictions.
    """
    image = tf.image.resize(image, image_dims)
    image = tf.image.convert_image_dtype(
        image=image, 
        dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    label_index, probs = inference.run_inference(image, MODEL)
    class_name = labels[label_index]
    return class_name, label_index, probs


def read_image(user_img):
    """Reads the image.
    Params:
        user_img: bytes
            The image from the camera.
    Returns:
        image: np.array
    """
    if user_img is not None:
        bytes_data = user_img.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)
        return img_tensor


def recipe_search(search_query: str,
                  max_recipes: int = 0):
    """Finds recipes from search_query.
    Params:
        search_query: str
            The search query.
        max_recipes: int
            The maximum number of recipes to return.
    Returns:
        recipes: list
    """
    NUTRITION.max_recipes = max_recipes
    recipes = NUTRITION.find_recipes(
        search_query)
    return recipes


def run_algorithm(button: bytes,
                  user_inputs: dict,
                  input_items: list,
                  input_item_nutritional_info: list):
    """Runs the SimpleShopping ALgorithm.
    Params:
        button: bytes
            The image from the camera.
        user_inputs: dict
            The user inputs.
        input_items: list
            The list of items.
        input_item_nutritional_info: list
            The list of nutritional info.
    Returns:
        user_inputs: dict
    """
    input_image = read_image(button)
    preds, _, _ = prediction(input_image)
    input_items.append(preds)
    recipes = recipe_search(input_items)
    nutritional_info = NUTRITION.nutritional_info(
        item=preds)
    input_item_nutritional_info.append(
        nutritional_info)
    user_inputs["items"] = {
        "Item Name": input_items,
        "Nutritional info": input_item_nutritional_info}
    user_inputs["possible_recipes"] = recipes
    return user_inputs
    

def website():
    """Creates the frontend."""
    user_inputs = {"items": None, # List dict of items.
                                  # {Item Names: None,
                                  #  Item Quantity: None,
                                  #  Item Description: None}
                   "possible_recipes": None } # List of dicts of recipes.
    st.set_page_config(layout="wide")
    original_title = '<p style="font-family:monospace; color:Black; font-size: 50px;">SimpleShopping</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px; color: #a1ae25;'>Camera Stream</h1>", 
            unsafe_allow_html=True)
        button = col1.camera_input("")
        input_items = []
        input_item_nutritional_info = []
        if button:
            user_inputs = run_algorithm(
                button, 
                user_inputs, 
                input_items, 
                input_item_nutritional_info)
    with col2:
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px; color: #a1ae25;'>Product List</h1>", 
            unsafe_allow_html=True)
        st.markdown(
            '<style background-color:green; </style>', 
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px; color: #a1ae25;'>Recipes</h1>", 
            unsafe_allow_html=True)
    public_url = ngrok.connect(port='80')
    print(f"The public URL is: {public_url}")

if __name__ == '__main__':
    website()