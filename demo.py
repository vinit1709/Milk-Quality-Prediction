import streamlit as st
import numpy as np
import joblib  # Assuming you're using a saved model

# Load the trained machine learning model
model = joblib.load('milk_quality_model_2.pkl')

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
