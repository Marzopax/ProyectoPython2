import io
import streamlit as st
import pandas as pd
from transformers import pipeline

# 1. Configuración del modelo local
@st.cache_resource
def cargar_modelo():
    # Usamos el modelo robusto local
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Cargamos el clasificador
classifier = cargar_modelo()

# Definimos las categorías que la IA va a evaluar (incluyendo la de descarte)
CATEGORIAS_IA = [
    "Depto. Hardware (computadoras, impresoras, monitores)", 
    "Depto. Software (programas, sistemas, licencias)", 
    "Depto. Redes (internet, wifi, cables, router)", 
    "No relacionado a informática (otros temas ajenos)"
]

st.title("Sistema Inteligente de Soporte Tecnico")
st.write("Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos y asignará el área correspondiente usando IA local.")

# Carga de archivo CSV
uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    # 2. FASE PANDAS (Limpieza y transformación - Clase 2)
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Filtros de limpieza inicial
        target_col = 'descripcion'
        if target_col in df.columns:
            df.dropna(subset=[target_col], inplace=True) # Elimina filas nulas
            df.drop_duplicates(subset=[target_col], inplace=True) # Elimina duplicados
            df[target_col] = df[target_col].astype(str).str.strip() # Limpia espacios
            
            st.write("### Datos normalizados con Pandas:", df.head())

            # 3. FASE IA (Inferencia local - Clase 6)
            if st.button("Clasificar requerimientos con IA"):
                with st.spinner("Clasificando localmente..."):
                    
                    def clasificar_texto(texto):
                        # La IA evalúa el texto frente a las 4 categorías
                        res = classifier(texto, CATEGORIAS_IA)
                        etiqueta_ganadora = res['labels'][0] # La opción con mayor probabilidad
                        
                        # Mapeo de la etiqueta de la IA a su categoría correspondiente
                        if "No relacionado" in etiqueta_ganadora:
                            return "Descartar"
                        elif "Hardware" in etiqueta_ganadora:
                            return "Depto. Hardware"
                        elif "Software" in etiqueta_ganadora:
                            return "Depto. Software"
                        elif "Redes" in etiqueta_ganadora:
                            return "Depto. Redes"
                        return "Descartar"

                    # Aplicación de la técnica IA sobre el DataFrame
                    df['Area_Asignada'] = df[target_col].apply(clasificar_texto)
                    
                    # --- FILTRADO CON PANDAS (Remover lo que no sirve) ---
                    # Eliminamos todas las filas donde la IA determinó que hay que "Descartar"
                    df = df[df['Area_Asignada'] != "Descartar"]
                    
                    st.success("¡Análisis completado! Se han filtrado y eliminado los registros no relacionados.")
                    
                    if not df.empty:
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
                        st.warning("⚠️ Todas las filas del archivo subido fueron descartadas por no estar relacionadas a informática.")
        else:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")
            
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
