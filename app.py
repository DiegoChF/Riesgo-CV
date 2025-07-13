import pdfkit
import tempfile
import os

# Campo para nombre del paciente
nombre_paciente = st.text_input("Nombre del paciente")

# Botón de predicción
if st.button("Calcular Riesgo"):

    if not nombre_paciente:
        st.warning("Por favor, introduce el nombre del paciente antes de continuar.")
    else:
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

        # Threshold definido
        threshold = 0.15

        if prob_alto >= threshold:
            pred = 'Alto'
        else:
            probs = modelo.predict_proba(entrada)[0]
            idx_bajo = list(le.classes_).index('Bajo')
            idx_mod = list(le.classes_).index('Moderado')
            pred = 'Bajo' if probs[idx_bajo] > probs[idx_mod] else 'Moderado'

        # Mostrar resultado
        st.subheader("Resultado del Riesgo Predicho:")
        st.markdown(f"<h2 style='color:purple'>{pred}</h2>", unsafe_allow_html=True)

        # Mensajes según riesgo
        if pred == 'Alto':
            recomendacion = "⚠️ Riesgo alto. Se recomienda evaluación con cardiología en el corto plazo."
            st.warning(recomendacion)
        elif pred == 'Moderado':
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

        # === GENERAR PDF ===
        html_content = f"""
        <h2>Resultado de Evaluación Cardiovascular</h2>
        <p><strong>Nombre del paciente:</strong> {nombre_paciente}</p>
        <p><strong>Riesgo predicho:</strong> {pred}</p>
        <p><strong>Recomendación:</strong> {recomendacion}</p>
        <hr>
        <p style='font-size:12px'>{disclaimer}</p>
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
            tmp_html.write(html_content.encode("utf-8"))
            tmp_html.flush()
            pdf_path = tmp_html.name.replace(".html", ".pdf")
            pdfkit.from_file(tmp_html.name, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Descargar Resultado en PDF", f, file_name=f"{nombre_paciente}_riesgo.pdf")
