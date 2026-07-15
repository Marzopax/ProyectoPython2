import io
import streamlit as st
import pandas as pd
from transformers import pipeline

# 1. Configuración del modelo local
# Esto carga el modelo en la memoria del servidor al iniciar
@st.cache_resource
def cargar_modelo():
    # Usamos un modelo de clasificación robusto que descarga localmente
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Cargamos el clasificador
classifier = cargar_modelo()
CATEGORIAS = ["Depto. Hardware", "Depto. Software", "Depto. Redes"]

st.title("🎫 Sistema Inteligente de Soporte Tecnico")
st.write("Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos y asignará el área correspondiente usando IA local.")

# Carga de archivo CSV
uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    # 2. FASE PANDAS (Limpieza y transformación - Clase 2)[cite: 1]
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Filtros de limpieza[cite: 1]
        target_col = 'descripcion'
        if target_col in df.columns:
            df.dropna(subset=[target_col], inplace=True) # Elimina filas nulas[cite: 1]
            df.drop_duplicates(subset=[target_col], inplace=True) # Elimina duplicados[cite: 1]
            df[target_col] = df[target_col].astype(str).str.strip() # Limpia espacios[cite: 1]
            
            st.write("### Datos normalizados con Pandas:", df.head())

            # 3. FASE IA (Inferencia local - Clase 6)[cite: 3]
            if st.button("🚀 Clasificar requerimientos con IA"):
                with st.spinner("Clasificando localmente..."):
                    
                    def clasificar_texto(texto):
                        res = classifier(texto, CATEGORIAS)
                        return res['labels'][0] # Asigna el área con mayor probabilidad[cite: 3]

                    # Aplicación de la técnica IA sobre el DataFrame[cite: 3]
                    df['Area_Asignada'] = df[target_col].apply(clasificar_texto)
                    
                    st.success("¡Análisis completado!")
                    st.dataframe(df)
                    
                    # Descarga de resultados en UTF-8 (con BOM para compatibilidad con Excel)
                    csv_buffer = io.BytesIO()
                    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Descargar CSV Clasificado",
                        csv_buffer.getvalue(),
                        "tickets_soporte_ia.csv",
                        "text/csv; charset=utf-8",
                    )
        else:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")
            
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
