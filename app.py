import streamlit as st
import pandas as pd
import pickle

# Load trained model and scaler
model = pickle.load(open("insulin_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Insulin Dosage Predictor", layout="centered")
st.title("💉 Insulin Dosage Predictor")
st.write("Predict insulin dosage based on patient health parameters.")

# User input fields
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=250, value=120)
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
age = st.slider("Age", 20, 90, 30)

# Extra categorical fields (if your training used them)
gender = st.selectbox("Gender", ["Male", "Female"])

# Helper functions to match training preprocessing
def get_age_group(age):
    if age < 40:
        return "Adult"
    else:
        return "Senior"

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Predict button
if st.button("Predict Insulin Dosage"):
    # Build raw input
    input_data = pd.DataFrame([{
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
        "Gender": gender,
        "AgeGroup": get_age_group(age),
        "BMI_Category": get_bmi_category(bmi)
    }])

    # One-hot encode exactly like training
    input_data = pd.get_dummies(input_data)

    # Ensure all features match training
    expected_features = scaler.feature_names_in_  # stored during fit
    for col in expected_features:
        if col not in input_data:
            input_data[col] = 0  # add missing cols
    input_data = input_data[expected_features]

    # Scale + predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Insulin Dosage: {round(prediction[0],2)} units")
