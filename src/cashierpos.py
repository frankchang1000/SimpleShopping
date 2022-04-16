
import streamlit as st

import numpy as np
from pyngrok import ngrok


st.set_page_config(layout="wide")
st.title('SimpleShopping') #Change the title here!


col1, col2 = st.columns(2)
col1.header("Input")
col1.camera_input("")


col2.header("Output")
public_url = ngrok.connect(port='80')
print (public_url)