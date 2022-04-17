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


print("Loading models...")
MODEL = tf.keras.models.load_model("data/simpleshopping-model")
print("Finished model loading...")
LABELS = training_utils.read_files("data/labels.txt")
print("Finished loading labels...")
NUTRITION = recipes_foods.NutritionInfo(
    consumer_key=file_reader.read_keys()["consumer_key"],
    consumer_secret=file_reader.read_keys()["consumer_secret"])
print("Finished nutrition info...")
ITEMS = file_reader.read_items_file(file_path="data/upc_corpus.csv")
print("Finished retrieving barcodes...")

def prediction(image: np.array,
               image_dims: tuple = (512, 512),
               labels: list = LABELS):
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
    print("Prediction label index: ", label_index)
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
    print("input image")
    input_image = read_image(button)
    print(input_image.shape)
    if barcode.scan_barcode(
        np.array(
            tf.squeeze(
                tf.image.rgb_to_grayscale(np.array(input_image)),
                axis=-1)),
        debug=True) != None:
        product_scanned = barcode.find_product(
            barcode.scan_barcode(input_image), 
            ITEMS)
        print(product_scanned)
        nutritional_info = NUTRITION.nutritional_info(
            product_scanned)
        print(nutritional_info)
        input_items.append(product_scanned)
        input_item_nutritional_info.append(nutritional_info)
        recipes = NUTRITION.find_recipes(input_items)
        user_inputs["items"] = {
            "item_name": input_items,
            "nutritional_info": input_item_nutritional_info}
        user_inputs["recipes"] = recipes
    else:
        preds, _, _ = prediction(input_image)
        print(preds)
        input_items.append(preds)
        recipes = NUTRITION.find_recipes(input_items)
        nutritional_info = NUTRITION.nutritional_info(
            item=preds)
        input_item_nutritional_info.append(
            nutritional_info)
        user_inputs["items"] = {
            "item_name": input_items,
            "nutritional_info": input_item_nutritional_info}
        user_inputs["possible_recipes"] = recipes
    print(user_inputs)
    return user_inputs
    

def website():
    """Creates the frontend."""
    user_inputs = {"items": None, # List dict of items.
                                  # {Item Names: None,
                                  #  Item Quantity: None,
                                  #  Item Description: None}
                   "possible_recipes": None } # List of dicts of recipes.
    st.set_page_config(layout="wide")

    #set up background colors
    set_bg_hack_url()

    #remove top banner
    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
        footer {visibility: hidden;}
        .css-vl8c1e {
            position: fixed;
            top: 0px;
            left: 0px;
            right: 0px;
            height: 0rem;
            background: linear-gradient(rgb(255, 255, 255) 25%, rgba(255, 255, 255, 0.5) 75%, transparent);
            backdrop-filter: blur(3px);
            z-index: 1000020;
            display: block;
        }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    col1, col2, col3 = st.columns([1.3 ,0.385,0.9])

    user_inputs = {
        "items": {"item_name": [''],
                  "item_quantity": None,
                  "nutritional_info": ['']},
        "possible_recipes": [{"recipe_name": '',
                             "recipe_description": '',
                             "calories_per_serving": 0.0,
                             "fat_per_serving": 0.0,
                             "protein_per_serving": 0.0,
                             "carbohydrate_per_serving": 0.0}]}
    with col1:
        st.image("data/logo.png")
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px; color: white;'>Camera Stream</h1>", 
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
            "<h1 style='text-align: center; font-size: 30px; color: black;'>Product List</h1>", 
            unsafe_allow_html=True)
        output = user_inputs['items']['item_name']
        print(user_inputs)
        item_nutritional = user_inputs['items']['nutritional_info']
        
        st.markdown('Predicted Item: ' + str(output)[2:-2])
        st.markdown('Nutritional Info: ' + str(item_nutritional))
    with col3:
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px; color: black;'>Recipes</h1>", 
            unsafe_allow_html=True)
        table_values_recipe = f"""
            <table>
                <tr>
                    <th>Recipe Name</th>
                    <th>Recipe Description</th>
                    <th>Calories Per Serving</th>
                    <th>Fat Per Serving</th>
                    <th>Protein Per Serving</th>
                    <th>Carbs Per Serving</th>
                </tr>
                <tr>
                    <th> {user_inputs['possible_recipes'][0]['recipe_name']} </th>
                    <th> {user_inputs['possible_recipes'][0]['recipe_description']} </th>
                    <th> {user_inputs['possible_recipes'][0]['calories_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][0]['fat_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][0]['protein_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][0]['carbohydrate_per_serving']} </th>
                </tr>
                <tr>
                    <th> {user_inputs['possible_recipes'][1]['recipe_name']} </th>
                    <th> {user_inputs['possible_recipes'][1]['recipe_description']} </th>
                    <th> {user_inputs['possible_recipes'][1]['calories_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][1]['fat_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][1]['protein_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][1]['carbohydrate_per_serving']} </th>
                </tr>
                <tr> 
                    <th> {user_inputs['possible_recipes'][2]['recipe_name']} </th>
                    <th> {user_inputs['possible_recipes'][2]['recipe_description']} </th>
                    <th> {user_inputs['possible_recipes'][2]['calories_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][2]['fat_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][2]['protein_per_serving']} </th>
                    <th> {user_inputs['possible_recipes'][2]['carbohydrate_per_serving']} </th>
                </tr>"""
        st.markdown(table_values_recipe, unsafe_allow_html=True)
        st.markdown("</table>", unsafe_allow_html=True)
        
        # Formet the table
        st.markdown(
            hide_streamlit_style = """
                <style>
                    #MainMenu {visibility: hidden;}
                    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
                    footer {visibility: hidden;}
                    .css-vl8c1e {
                        position: fixed;
                        top: 0px;
                        left: 0px;
                        right: 0px;
                        height: 0rem;
                        background: linear-gradient(rgb(255, 255, 255) 25%, rgba(255, 255, 255, 0.5) 75%, transparent);
                        backdrop-filter: blur(3px);
                        z-index: 1000020;
                        display: block;
                    }
                </style>
                """,
            unsafe_allow_html=True)

    public_url = ngrok.connect(port='80')
    print("The public URL is: {public_url}")


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.ibb.co/V3VT5Mw/Untitled-drawing.png");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


if __name__ == '__main__':
    website()