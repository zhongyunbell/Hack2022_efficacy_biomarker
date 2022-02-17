import streamlit as st
import matplotlib.pyplot as plt
import shapml

st.set_page_config(page_title = 'Treatment efficacy biomarkers', page_icon= '⚕️', layout="wide")

# Put things in sidebar
file=st.sidebar.file_uploader('load file')
if type(file) != type(None):
    st.sidebar.write(file.name)
    section = st.sidebar.radio('Section:',
            ['Home',
            'Exploratory data analysis',
            'SHAP analysis',
            'Potential treatment effect modifiers'], index=3)
else:
    section = 'Home'

if section != 'Home': 
    st.header(section)    
if section == 'Home':
    exec(open("pages/home.py").read())

elif section == 'Exploratory data analysis':
    exec(open("pages/EDA.py").read())

elif section == 'SHAP analysis':
    exec(open("pages/shap_analysis.py").read())

elif section =='Potential treatment effect modifiers':
    exec(open("pages/treatment_effect_modifiers.py").read())
