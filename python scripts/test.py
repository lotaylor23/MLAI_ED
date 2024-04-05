import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import geopandas as gpd
from branca.colormap import linear
import folium
from streamlit_folium import st_folium
import os, glob

import streamlit.components.v1 as components  # Import Streamlit



st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](testload)
-'file:///C:/Users/lstaylor/Desktop/final%20project%20data/5yr%20basicA%20vs%20A2.html')

    
"""
)
data = pd.read_csv("C:/Users/lstaylor/Desktop/final project data/report (5).csv")
image = Image.open(r"C:\Users\lstaylor\Desktop\final project data\5yr basicA.jpg")
geojson = gpd.read_file("C:/Users/lstaylor/Desktop/final project data/Current_Districts_2023.geojson")
colormap = linear.YlGn_09.scale(
   data["Number Tested"].min(),
    data["Number Tested"].max())
nt_dict = data.set_index('Org Id')['Number Tested']

nt_dict#['TX']
m = folium.Map([31.5, -100], zoom_start=5.56)
folium.Choropleth(
    geo_data=geojson,
    name = "geojson",
    style_function = lambda x:style,
    fill_opacity=0.3,
    line_weight=2,
    data=data,
    columns=['Org Id','Number Tested'],
    key_on='feature.properties.DISTRICT_N',
).add_to(m)

folium.GeoJson(geojson,
               zoom_on_click=True,
               name='data'
              ).add_to(m)
folium.LayerControl().add_to(m)
st_data=st_folium(m)
st.image(image)
