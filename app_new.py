import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from transformers import pipeline

MIN_CARACTERES = 10
CATEGORIAS = ["Depto. Hardware", "Depto. Software", "Depto. Redes"]


@st.cache_resource
def cargar_modelo():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
    )


classifier = cargar_modelo()


def es_texto_valido(texto):
    texto = str(texto).strip()
    if len(texto) < MIN_CARACTERES:
        return False
    if not re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]", texto):
        return False
    return True


def clasificar_lote(textos, umbral):
    areas = [None] * len(textos)
    confianzas = [None] * len(textos)
    textos_validos = []
    indices_validos = []

    for i, texto in enumerate(textos):
        if not es_texto_valido(texto):
            areas[i] = "Invalido"
            confianzas[i] = 0.0
        else:
            textos_validos.append(texto)
            indices_validos.append(i)

    if textos_validos:
        preds = classifier(textos_validos, CATEGORIAS)
        if isinstance(preds, dict):
            preds = [preds]

        for idx, pred in zip(indices_validos, preds):
            score = pred["scores"][0]
            confianzas[idx] = round(score, 4)
            if score < umbral:
                areas[idx] = "Invalido"
            else:
                areas[idx] = pred["labels"][0]

    return areas, confianzas


st.title("Sistema Inteligente de Soporte Tecnico")
st.write(
    "Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos "
    "y asignará el área correspondiente usando IA local."
)

uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")

        target_col = "descripcion"
        if target_col in df.columns:
            df.dropna(subset=[target_col], inplace=True)
            df.drop_duplicates(subset=[target_col], inplace=True)
            df[target_col] = df[target_col].astype(str).str.strip()

            st.write("### Datos normalizados con Pandas:", df.head())

            umbral = st.slider(
                "Umbral de confianza",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Por debajo de este valor la descripción se marca como Invalido.",
            )

            if st.button("Clasificar requerimientos"):
                with st.spinner("Clasificando localmente..."):
                    areas, confianzas = clasificar_lote(df[target_col].tolist(), umbral)

                    df["Area_Asignada"] = areas
                    df["Confianza"] = confianzas
                    df["fecha_clasificacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df["estado"] = "pendiente"

                st.success("¡Análisis completado!")
                st.write("### Resumen por área")
                resumen = df["Area_Asignada"].value_counts().reset_index()
                resumen.columns = ["Area", "Cantidad"]
                st.dataframe(resumen, hide_index=True)
                st.bar_chart(resumen.set_index("Area"))

                st.write("### Resultados detallados")
                st.dataframe(df)

                csv_buffer = io.BytesIO()
                df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
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
