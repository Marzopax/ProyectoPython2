import streamlit as st
import pandas as pd
from transformers import pipeline

# ============================================================
# CARGA DEL MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier = cargar_modelo()

# ============================================================
# CATEGORÍAS DE CLASIFICACIÓN (Optimizadas para Zero-Shot)
# ============================================================

# Se usan etiquetas cortas para que el modelo no se sature de texto.
# La opción "Otros (No Informática)" actuará como trampa para los tickets basura.
CATEGORIAS_LABELS = [
    "Falla de Hardware (Equipos físicos)", 
    "Falla de Software (Programas)", 
    "Otros (No Informática / Infraestructura / Redes)"
]

# ============================================================
# INTERFAZ
# ============================================================

st.title("Sistema Inteligente de Soporte (Filtro Estricto)")
st.write("Subí un CSV. El sistema conservará ÚNICAMENTE los requerimientos de Hardware y Software.")

# ============================================================
# SLIDERS DE CALIBRACIÓN
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    UMBRAL_CONFIANZA = st.slider("Umbral confianza (normales)", 0.0, 1.0, 0.35, 0.05)
with col2:
    UMBRAL_CORTO = st.slider("Umbral confianza (cortas)", 0.0, 1.0, 0.20, 0.05)
with col3:
    LONGITUD_CORTA = st.slider("Umbral de longitud (caracteres)", 0, 50, 10, 1)

# ============================================================
# PROCESAMIENTO
# ============================================================

def reparar_mojibake(texto):
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto

uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')

        target_col = 'descripcion'
        if target_col in df.columns:

            # -------------------- LIMPIEZA INICIAL PANDAS --------------------
            df[target_col] = df[target_col].astype(str).apply(reparar_mojibake)
            df.dropna(subset=[target_col], inplace=True)
            df.drop_duplicates(subset=[target_col], inplace=True)
            df[target_col] = df[target_col].astype(str).str.strip()
            df = df[df[target_col].str.len() > 3]

            st.write("### Datos normalizados inicialmente:", df.head())

            # -------------------- CLASIFICACIÓN CON IA --------------------
            if st.button("🚀 Clasificar y Filtrar"):
                with st.spinner("Clasificando localmente..."):

                    def clasificar_texto(texto):
                        res = classifier(
                            texto,
                            CATEGORIAS_LABELS,
                            multi_label=False, # CORRECCIÓN: Obliga a las categorías a competir
                            hypothesis_template="Este reclamo de soporte técnico trata sobre: {}."
                        )
                        mejor_hipotesis = res['labels'][0]
                        mejor_score = res['scores'][0]
                        
                        # Simplificación para el DataFrame
                        if "Hardware" in mejor_hipotesis:
                            categoria = "Hardware"
                        elif "Software" in mejor_hipotesis:
                            categoria = "Software"
                        else:
                            categoria = "Otros"
                            
                        return categoria, mejor_score

                    resultados = df[target_col].apply(clasificar_texto)
                    df['Area_Asignada'] = resultados.apply(lambda x: x[0])
                    df['Confianza'] = resultados.apply(lambda x: round(x[1], 3))
                    df['Longitud'] = df[target_col].str.len()

                    st.session_state['df_clasificado'] = df

        # -------------------- FILTRADO ESTRICTO PANDAS --------------------
        if 'df_clasificado' in st.session_state:
            df_clasificado = st.session_state['df_clasificado']
            total_antes = len(df_clasificado)

            # 1. Filtro de calidad (elimina descripciones basura por baja confianza)
            es_corta = df_clasificado['Longitud'] < LONGITUD_CORTA
            pasa_corta = es_corta & (df_clasificado['Confianza'] >= UMBRAL_CORTO)
            pasa_normal = ~es_corta & (df_clasificado['Confianza'] >= UMBRAL_CONFIANZA)
            df_filtrado = df_clasificado[pasa_corta | pasa_normal].copy()

            # 2. ELIMINACIÓN ESTRICTA CON PANDAS
            # Conservamos únicamente Hardware y Software. Lo que caiga en "Otros" desaparece.
            df_filtrado = df_filtrado[df_filtrado['Area_Asignada'].isin(["Hardware", "Software"])].copy()

            descartados = total_antes - len(df_filtrado)

            st.success(f"¡Filtrado completado! Se descartaron {descartados} registro(s) ajenos a los departamentos principales.")
            
            if not df_filtrado.empty:
                st.dataframe(df_filtrado[['descripcion', 'Area_Asignada', 'Confianza']])
                csv = df_filtrado.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 Descargar CSV Filtrado", csv, "tickets_filtrados.csv", "text/csv")
            else:
                st.warning("No quedó ningún registro de Hardware o Software después del filtro.")

        elif target_col not in df.columns:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
