# Note: analysis is defined upstream from app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from shapml import binary_classification as BC
from shapml import utils

st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Covariate impacts on exposure')
st.write('(log-odds)')
out = analysis.shap_exposure_impacts(exposure_var=analysis.exposure_var)
col1, col2 =st.columns([0.3, 0.7])
with col1:
    st.dataframe(out[['covariate','mean impact']])
with col2:
    # plot on covariate impact
    fig_covariate_impact = plt.figure(figsize=(10, 4))
    sns.barplot(x='covariate', y='mean impact', data=out)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_covariate_impact)

col1, col2 =st.columns([0.4, 0.6])
with col1:
    # shap interaction summary
    st.subheader('Summary of interactions with drug exposure')
    # feature_interaction = st.radio('Pick a factor for interaction summary', ['CTR1', 'MADCAM1'])
    fig_interactions_summary = analysis.shap_interaction_summary(feature=analysis.exposure_var, figsize=(15,15))
    st.pyplot(fig_interactions_summary)
    
with col2:
    # select factor that interact with exposure 
    st.subheader('Covariate interactions with drug exposure:')
    interacting_factor = st.selectbox('Select Interacting Factor', out.covariate.values)
    fig_interactions = analysis.dependence_plot(analysis.exposure_var, interacting_factor)
    st.pyplot(fig_interactions)

# Abid: insert your code here:
