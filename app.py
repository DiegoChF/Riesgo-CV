import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, label encoder, and features
try:
    model = joblib.load('modelo_rf.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please upload 'modelo_rf.pkl', 'label_encoder.pkl', and 'features.pkl'.")
    st.stop()

st.title("Predicción de Riesgo Cardiovascular")

st.write("""
Esta aplicación predice el riesgo cardiovascular (Alto, Bajo, Moderado) basado en los factores de riesgo ingresados.
""")

# Input fields
edad = st.slider("Edad", 18, 100, 50)
sexo = st.selectbox("Sexo", options=["Masculino", "Femenino"])
pas = st.slider("Presión Arterial Sistólica (PAS)", 80, 200, 120)
colesterol_total = st.slider("Colesterol Total", 100, 400, 200)
hdl = st.slider("HDL", 20, 100, 50)
tabaquismo = st.selectbox("Tabaquismo", options=["No", "Sí"])
diabetes = st.selectbox("Diabetes", options=["No", "Sí"])

# Map input to model features
sexo_encoded = 1 if sexo == "Masculino" else 0
tabaquismo_encoded = 1 if tabaquismo == "Sí" else 0
diabetes_encoded = 1 if diabetes == "Sí" else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame([[edad, sexo_encoded, pas, colesterol_total, hdl, tabaquismo_encoded, diabetes_encoded]],
                          columns=features)


# Make prediction
if st.button("Predecir Riesgo"):
    # Ensure input data has the same columns and order as the training data
    input_data = input_data[features]

    probs = model.predict_proba(input_data)[0]
    predicted_class_index = np.argmax(probs)
    predicted_risk = le.classes_[predicted_class_index]

    st.subheader("Resultado de la Predicción:")

    # Apply threshold logic from the notebook (optional, based on user preference)
    # For simplicity, we'll use the standard predict for this Streamlit app
    # If the user wants the threshold logic, it can be added here.

    st.write(f"El riesgo cardiovascular predicho es: **{predicted_risk}**")

    st.subheader("Probabilidades por Categoría:")
    probs_df = pd.DataFrame([probs], columns=le.classes_)
    st.dataframe(probs_df)

# Add information about the threshold if desired
# st.info(f"The model uses a threshold of {th} for the 'Alto' class.")
