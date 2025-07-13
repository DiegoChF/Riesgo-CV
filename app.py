import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo y codificador
modelo = joblib.load('modelo_rf.pkl')
le = joblib.load('label_encoder.pkl')

# Título
st.title("Predicción de Riesgo Cardiovascular")
st.markdown("Calcular el Riesgo Cardiovascular a 10 años")

# Formulario
st.subheader("Introduce los datos del paciente:")

edad = st.number_input("Edad", min_value=20, max_value=100, step=1)
sexo = st.radio("Sexo", ["Masculino", "Femenino"])
pas = st.number_input("Presión Arterial Sistólica (mmHg)", min_value=40, max_value=250, step=1)
col_total = st.number_input("Colesterol Total (mg/dL)", min_value=60, max_value=400, step=1)
hdl = st.number_input("Colesterol HDL (mg/dL)", min_value=20, max_value=150, step=1)
tabaquismo = st.radio("¿Fuma actualmente?", ["Sí", "No"])
diabetes = st.radio("¿Tiene diabetes?", ["Sí", "No"])

# Botón de predicción
if st.button("Calcular Riesgo"):

    # Convertir entradas a formato esperado por el modelo
    entrada = pd.DataFrame([[
        edad,
        1 if sexo == "Masculino" else 0,
        pas,
        col_total,
        hdl,
        1 if tabaquismo == "Sí" else 0,
        1 if diabetes == "Sí" else 0
    ]], columns=[
        'Edad', 'Sexo', 'PAS', 'Colesterol_total', 'HDL', 'Tabaquismo', 'Diabetes'
    ])

    # Probabilidad de clase "Alto"
    idx_alto = list(le.classes_).index('Alto')
    prob_alto = modelo.predict_proba(entrada)[0][idx_alto]

    # Threshold definido durante entrenamiento
    threshold = 0.15

    if prob_alto >= threshold:
        pred = 'Alto'
    else:
        # Comparar entre Bajo y Moderado
        probs = modelo.predict_proba(entrada)[0]
        idx_bajo = list(le.classes_).index('Bajo')
        idx_mod = list(le.classes_).index('Moderado')
        pred = 'Bajo' if probs[idx_bajo] > probs[idx_mod] else 'Moderado'

    # Mostrar resultado
    st.subheader("Resultado del Riesgo Predicho:")
    st.markdown(f"<h2 style='color:purple'>{pred}</h2>", unsafe_allow_html=True)

    # Mensajes según riesgo
            if pred = 'Alto':
                recomendacion = "⚠️ Riesgo alto. Se recomienda evaluación con cardiología en el corto plazo."
                st.warning(recomendacion)
            elif pred = 'Moderado':
                recomendacion = "ℹ️ Riesgo moderado. Sugiere evaluación médica en el mediano plazo."
                st.info(recomendacion)
            else:
                recomendacion = "✅ Riesgo bajo. Acuda a su control médico anual para seguimiento."
                st.success(recomendacion)
    
            # Descargo de responsabilidad
            disclaimer = ("📝 **Importante:** Esta herramienta no reemplaza una evaluación médica completa. "
                          "Los resultados deben ser interpretados por profesionales de salud.")
    
            st.markdown("---")
            st.markdown(disclaimer)    
