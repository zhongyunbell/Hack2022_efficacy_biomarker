# Note: analysis_name, full_analysis_name, expsosure, target, defined upstream in app.py
import streamlit as st
import pandas as pd
import shapml.binary_classification as BC
import copy

# st.write(data_path)
# df_orig = pd.read_csv(data_path)
data_path = st.file_uploader('Upload data: ')
if type(data_path) != type(None):
    df_orig = pd.read_csv(data_path)
    st.dataframe(df_orig)
    potential_features = df_orig.columns.tolist()
    potential_features.remove(target)
    # target = st.selectbox(label='Outcome variable', options=df_orig.columns)

    analysis_form=st.form(key='analyze data', clear_on_submit=False)
    with analysis_form:
        # select target variable
        final_model_features = st.multiselect(options=potential_features, default=potential_features, label='Select model features')
        # exposure_var = st.selectbox('Exposure Variable', options=potential_features)
        
        submit=st.form_submit_button(label='Run analysis')
        if submit:
            st.write("Running analysis *incorporate a status bar")
            mdl_terms = copy.deepcopy(final_model_features)
            mdl_terms.append(target)
            analysis=BC.xgb_shap(df=df_orig[mdl_terms],
                        target=target, exposure_var=exposure_var,
                        max_evals=5, bootstrap_iterations=100, n_folds_SHAP=5,
                        outputs_dir=outputs_dir)# optional parameters set for the sake of computations efficiency
            analysis.tune_model()
            analysis.shap_summary_plots()
            analysis.save(full_analysis_name, include_time=False)
            # st.write(analysis.hyperparams)
            st.write("Analysis completed!")

            initial_analysis_complete=True

if initial_analysis_complete:
    if st.button(label='Show analysis results'):
        from streamlit.script_runner import RerunException
        # do other stuff
        raise RerunException()