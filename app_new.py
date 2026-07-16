import streamlit as st
import pandas as pd
from transformers import pipeline

# ============================================================
# CARGA DEL MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    # @st.cache_resource evita recargar el modelo en cada interacción.
    # mDeBERTa-v3-base-mnli-xnli: modelo NLI multilingüe usado para zero-shot.
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier = cargar_modelo()

# ============================================================
# CATEGORÍAS DE CLASIFICACIÓN (Modificado)
# ============================================================

# Añadimos "Invalido" con una descripción semántica fuerte para capturar
# todo aquello ajeno a los tres departamentos principales de informática.
CATEGORIAS_DESC = {
    "Hardware": "Falla física de un equipo o dispositivo: impresora, computadora, monitor, teclado, mouse, cable, "
                "escáner o periférico que no enciende, no responde, hace ruido, se traba o está roto físicamente",
    "Software": "Falla de un programa o aplicación instalada: error al abrir, se cierra solo, licencia vencida, "
                "actualización fallida, mensaje de error en pantalla, problema al instalar un programa",
    "Redes": "Falla de conectividad: sin acceso a internet, wifi que no conecta, VPN caída, conexión lenta o "
             "intermitente, el router o módem sin luces o sin señal",
    "Invalido": "Cualquier texto, consulta, saludo o mensaje que sea totalmente ajeno a la informática, soporte técnico, "
                "computadoras, programas o redes (por ejemplo: comentarios sobre comida, compras de oficina, mobiliario, "
                "picaportes rotos, charlas generales o frases sin sentido)"
}
CATEGORIAS_LABELS = list(CATEGORIAS_DESC.keys())       # Nombres cortos (para mostrar en el resultado)
CATEGORIAS_HIPOTESIS = list(CATEGORIAS_DESC.values())  # Descripciones largas (para alimentar al modelo)

# ============================================================
# INTERFAZ: TÍTULO Y DESCRIPCIÓN
# ============================================================

st.title("Sistema Inteligente de Soporte")
st.write("Subí un CSV, el sistema lo limpiará y organizará.")
st.write("Con los sliders podés controlar los umbrales y longitud para filtrar descripción.")

# ============================================================
# SLIDERS DE CALIBRACIÓN
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    UMBRAL_CONFIANZA = st.slider(
        "Umbral confianza (normales)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Aplica a descripciones iguales o más largas que el umbral de longitud."
    )
with col2:
    UMBRAL_CORTO = st.slider(
        "Umbral confianza (cortas)",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Aplica a descripciones más cortas que el umbral de longitud."
    )
with col3:
    LONGITUD_CORTA = st.slider(
        "Umbral de longitud (caracteres)",
        min_value=0, max_value=50, value=10, step=1,
        help="Descripciones con menos caracteres que este valor se consideran 'cortas' y usan el umbral de confianza correspondiente."
    )

# ============================================================
# CARGA DEL ARCHIVO CSV
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

            # -------------------- LIMPIEZA CON PANDAS --------------------
            df[target_col] = df[target_col].astype(str).apply(reparar_mojibake)
            df.dropna(subset=[target_col], inplace=True)
            df.drop_duplicates(subset=[target_col], inplace=True)
            df[target_col] = df[target_col].astype(str).str.strip()
            df = df[df[target_col].str.len() > 3]

            st.write("### Datos normalizados con Pandas:", df.head())

            # -------------------- CLASIFICACIÓN CON IA --------------------
            if st.button("🚀 Clasificar requerimientos con IA"):
                with st.spinner("Clasificando localmente..."):

                    def clasificar_texto(texto):
                        res = classifier(
                            texto,
                            CATEGORIAS_HIPOTESIS,
                            multi_label=True,
                            hypothesis_template="Este reclamo de soporte técnico trata sobre: {}."
                        )
                        mejor_hipotesis = res['labels'][0]
                        mejor_score = res['scores'][0]
                        segunda_hipotesis = res['labels'][1]
                        segundo_score = res['scores'][1]
                        
                        idx = CATEGORIAS_HIPOTESIS.index(mejor_hipotesis)
                        categoria = CATEGORIAS_LABELS[idx]
                        
                        idx2 = CATEGORIAS_HIPOTESIS.index(segunda_hipotesis)
                        categoria_alternativa = CATEGORIAS_LABELS[idx2]
                        
                        gap = round(mejor_score - segundo_score, 3)
                        return categoria, mejor_score, categoria_alternativa, gap

                    resultados = df[target_col].apply(clasificar_texto)
                    df['Area_Asignada'] = resultados.apply(lambda x: x[0])
                    df['Confianza'] = resultados.apply(lambda x: round(x[1], 3))
                    df['Area_Alternativa'] = resultados.apply(lambda x: x[2])
                    df['Ambiguedad'] = resultados.apply(lambda x: x[3])
                    df['Longitud'] = df[target_col].str.len()

                    st.session_state['df_clasificado'] = df

        # -------------------- FILTRADO Y RESULTADOS --------------------
        if 'df_clasificado' in st.session_state:
            df_clasificado = st.session_state['df_clasificado']
            total_antes = len(df_clasificado)

            # 1. Filtro por umbrales de confianza configurados en sliders
            es_corta = df_clasificado['Longitud'] < LONGITUD_CORTA
            pasa_corta = es_corta & (df_clasificado['Confianza'] >= UMBRAL_CORTO)
            pasa_normal = ~es_corta & (df_clasificado['Confianza'] >= UMBRAL_CONFIANZA)

            df_filtrado = df_clasificado[pasa_corta | pasa_normal].copy()
            
            # 2. Forzar que cualquier registro que haya caído en "Invalido" por la IA
            # o que no pertenezca a Hardware, Software o Redes se marque como Invalido.
            # (Adicionalmente, si el score de confianza es demasiado bajo, también podemos reasignarlo)
            df_filtrado.loc[~df_filtrado['Area_Asignada'].isin(["Hardware", "Software", "Redes"]), 'Area_Asignada'] = "Invalido"

            descartados = total_antes - len(df_filtrado)

            st.success(f"¡Análisis completado! {descartados} registro(s) descartado(s) por baja confianza.")
            st.dataframe(df_filtrado)

            csv = df_filtrado.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")

        elif target_col not in df.columns:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
