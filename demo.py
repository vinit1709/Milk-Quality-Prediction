import streamlit as st
import numpy as np
import joblib
import os

# Check if the model file exists and load it
if os.path.exists("milk_quality_model_2.pkl"):
    try:
        model = joblib.load("milk_quality_model_2.pkl")
    except ModuleNotFoundError as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()  # Stop execution if model loading fails
else:
    st.error("Model file not found. Please check the file path and try again.")

# Title of the web app
st.title("Milk Quality Prediction")

# Input fields for user input
fat = st.number_input("Fat:", min_value=0.0, format="%.2f")
snf = st.number_input("SNF:", min_value=0.0, format="%.2f")
protein = st.number_input("Protein:", min_value=0.0, format="%.2f")
lr = st.number_input("LR:", min_value=0.0, format="%.2f")
celsius = st.number_input("Celsius:", min_value=0.0, format="%.2f")
oil = st.number_input("Oil:", min_value=0.0, format="%.2f")
water = st.number_input("Water:", min_value=0.0, format="%.2f")

# Button for prediction
if st.button("Predict"):
    try:
        # Prepare the feature array for prediction
        features = np.array([[fat, snf, protein, lr, celsius, oil, water]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display the prediction result
        st.success(f"Predicted Milk Quality: {prediction[0]}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
