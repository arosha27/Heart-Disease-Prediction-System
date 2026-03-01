

import streamlit as st
import os
import pandas as pd
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="",
    layout="centered"
)

# ---------------------- CSS for Styling ----------------------
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
        color: #333333;
    }
    /* Input fields styling */
    div[data-baseweb="select"] > div > div, input {
        background-color: #FFF3E0;
        border-radius: 10px;
    }
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #FF6B6B;
        color:white;
        font-size:20px;
        border-radius:10px;
        height:50px;
        width:200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Load Model ----------------------
model_path = os.path.join(os.path.dirname(__file__), "CatBoostModel.cbm")
model = CatBoostClassifier()
model.load_model(model_path)

# ---------------------- Header ----------------------
st.markdown(
    "<h1 style='text-align: center; color:#FF3E3E;'>Heart Disease Prediction Using CatBoost</h1>",
    unsafe_allow_html=True
)

# ---------------------- Input Fields ----------------------
with st.expander("Patient Information", expanded=True):
    id_input = st.number_input("ID", min_value=1, max_value=60000, value=1)
    Age = st.number_input("Age", min_value=1, max_value=100, value=25)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Sex = 1 if Sex == "Male" else 0
    BP = st.number_input("Blood Pressure", min_value=10, max_value=250, value=120)
    Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    FBS_over_120 = st.selectbox("FBS over 120", [0, 1])
    Max_HR = st.number_input("Max Heart Rate", min_value=10, max_value=250, value=150)
    Exercise_angina = st.selectbox("Exercise Angina", [0, 1])
    ST_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    Number_of_vessels_fluro = st.selectbox("Number of vessels fluro", [0, 1, 2, 3])
    Chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    EKG_results = st.selectbox("EKG Results", [0, 1, 2])
    Thallium = st.selectbox("Thallium", [3, 6, 7])
    Slope_of_ST = st.selectbox("Slope of ST", [1, 2, 3])

# ---------------------- Prediction ----------------------
if st.button("Predict"):

    # Define all model feature columns
    feature_columns = [
        'id', 'Age', 'Sex', 'BP', 'Cholesterol', 'FBS over 120', 'Max HR',
        'Exercise angina', 'ST depression', 'Number of vessels fluro',
        'Chest pain type_1', 'Chest pain type_2', 'Chest pain type_3', 'Chest pain type_4',
        'EKG results_0', 'EKG results_1', 'EKG results_2',
        'Thallium_3', 'Thallium_6', 'Thallium_7',
        'Slope of ST_1', 'Slope of ST_2', 'Slope of ST_3'
    ]
    
    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Assign numeric features
    df.at[0, 'id'] = id_input
    df.at[0, 'Age'] = Age
    df.at[0, 'Sex'] = Sex
    df.at[0, 'BP'] = BP
    df.at[0, 'Cholesterol'] = Cholesterol
    df.at[0, 'FBS over 120'] = FBS_over_120
    df.at[0, 'Max HR'] = Max_HR
    df.at[0, 'Exercise angina'] = Exercise_angina
    df.at[0, 'ST depression'] = ST_depression
    df.at[0, 'Number of vessels fluro'] = Number_of_vessels_fluro

    # Assign one-hot encoded features
    df.at[0, f'Chest pain type_{Chest_pain_type}'] = 1
    df.at[0, f'EKG results_{EKG_results}'] = 1
    df.at[0, f'Thallium_{Thallium}'] = 1
    df.at[0, f'Slope of ST_{Slope_of_ST}'] = 1

    # ---------------------- Prediction ----------------------
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader("Prediction Result")
    st.write("Prediction probability (No Disease / Disease):", prediction_proba[0])
    if prediction[0] == 1:
        st.error("Patient diagnosed with Heart Disease")
    else:
        st.success("Patient is Healthy")

    # ---------------------- SHAP Feature Contributions ----------------------
    st.subheader("Feature Contribution (SHAP Values)")

    # Calculate SHAP values
    shap_values = model.get_feature_importance(Pool(df), type="ShapValues")
    shap_values_for_row = shap_values[0, :-1]  # last column is base value

    # Color coding: red → pushes toward disease, green → pushes toward healthy
    colors = ["green" if val < 0 else "red" for val in shap_values_for_row]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df.columns, shap_values_for_row, color=colors)
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title("Feature Contribution for This Patient")
    st.pyplot(fig)

    # Optional: show input DataFrame
    st.subheader("Input Features Sent to Model")
    st.dataframe(df)