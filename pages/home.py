#Import libraries
import streamlit as st
# import numpy as np
#import cv2
from  PIL import Image #, ImageEnhance


image1 = Image.open(r'figs/logo.png') #Brand logo image


#Create two columns with different width
col1, col2 = st.columns( [0.9, 0.1])
with col1:               # To display the header text using css style
    original_title = '<p style="font-family:Courier; color:Black; font-weight: bold;font-size:120px;">Precisio</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    
with col2:               # To display brand logo
    st.image(image1,  width=100)
    

url = "Identify biomarkers for treatment efficacy"
st.markdown(f'<p style="text-align:center; background-color:LightSkyBlue;color:Black;font-size:30px;border-radius:1%;">{url}</p>', unsafe_allow_html=True)

image2 = Image.open(r'figs/Hack_fig.png') #Main image 
st.image(image2)

#Add a header and expander in side bar
#st.sidebar.markdown('My First Photo Converter App', unsafe_allow_html=True)
st.write("""This app utilizes clinical trial data to identify predictive biomarkers of therapeutic efficacy and assesses their predictive utility""")
# with st.sidebar.expander("About the App"):
#      st.write("""
#         It is paramount to identify predictive biomarkers for therapeutic efficacy  \n  \nWe developed this app to identify predictive biomarkers and assesses their predictive utility from clinical trial data
    #  """)
        