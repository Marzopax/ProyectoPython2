import streamlit as st
import pandas as pd
import requests

# Configuración de la API de Hugging Face
HF_TOKEN = "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"
MODEL_URL = "https://api-inference.huggingface.co/models/Recognai/bert-base-spanish-wwm-cased-xnli"
CATEGORIAS = ["Hardware / Equipos", "Software / Aplicaciones", "Redes / Internet"]

st.title("🎫 Sistema de Triaje de Soporte Inteligente")
st.write("Subí un CSV con tus requerimientos de soporte. El sistema limpiará los datos y usará un LLM para asignar el área encargada.")

# Carga de archivo
uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    # 1. FASE PANDAS (Limpieza y transformación)
    try:
        # Manejo de errores de codificación para archivos en español
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Técnicas de limpieza (Clase 2)
        st.write("Columnas detectadas:", list(df.columns))
        col_name = 'descripcion' # Asegúrate que tu CSV tenga esta columna
        
        if col_name in df.columns:
            df.dropna(subset=[col_name], inplace=True)       # Elimina nulos
            df.drop_duplicates(subset=[col_name], inplace=True) # Elimina duplicados[cite: 1]
            df[col_name] = df[col_name].astype(str).str.strip() # Normaliza texto
            
            st.dataframe(df.head())

            # 2. FASE LLM (Inferencia)
            if st.button("🚀 Procesar e Integrar con IA"):
                with st.spinner("Conectando con el LLM..."):
                    def obtener_clasificacion(texto):
                        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                        payload = {"inputs": texto, "parameters": {"candidate_labels": CATEGORIAS}}
                        response = requests.post(MODEL_URL, headers=headers, json=payload)
                        return response.json()['labels'][0]

                    # Aplicar clasificación a cada fila[cite: 1]
                    df['Area_Asignada'] = df[col_name].apply(obtener_clasificacion)
                    
                    st.success("¡Pipeline completado!")
                    st.dataframe(df)

                    # Descarga
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")
        else:
            st.error(f"El CSV debe tener una columna llamada '{col_name}'.")
            
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")