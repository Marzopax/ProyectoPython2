# Sistema Inteligente de Soporte (Procesamiento Local)

Clasifica automáticamente requerimientos de soporte técnico (Hardware, Software, Redes) usando IA local, sin depender de APIs externas. Filtra descripciones irrelevantes usando umbrales de confianza configurables.

## Deploy

App en vivo: **[proyectopython2-dzzfrfgbsifwprseca4yd4.streamlit.app](https://proyectopython2-dzzfrfgbsifwprseca4yd4.streamlit.app)**

## Stack

| Tecnología | Uso |
|---|---|
| [Streamlit](https://streamlit.io/) | Interfaz web e interactividad (sliders, carga de archivos) |
| [Pandas](https://pandas.pydata.org/) | Limpieza y transformación de datos CSV |
| [Transformers](https://huggingface.co/docs/transformers) (Hugging Face) | Pipeline de zero-shot classification |
| [PyTorch](https://pytorch.org/) | Backend de inferencia del modelo |
| [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | Modelo NLI multilingüe (incluye español) |

## Requisitos

```bash
pip install streamlit pandas transformers torch
```

> Nota: la primera ejecución descarga el modelo `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (~500 MB). Requiere conexión a internet solo esa primera vez; luego queda cacheado localmente.

## Ejecución

```bash
streamlit run app_new.py
```

Se abre automáticamente en el navegador (`http://localhost:8501`).

## Formato del CSV de entrada

El archivo debe contener obligatoriamente una columna llamada **`descripcion`**. Columnas adicionales (ej. `id`) se preservan sin problema.

| id | descripcion |
|---|---|
| 1 | La impresora no toma las hojas y hace un ruido extraño. |
| 2 | Necesito que me reinstalen la licencia de Excel. |

Codificación soportada: UTF-8 o Latin-1 (detección automática).

## Flujo de uso

1. **Subir CSV** con el botón de carga de archivo.
2. El sistema limpia automáticamente los datos:
   - Elimina filas sin descripción.
   - Elimina descripciones duplicadas.
   - Recorta espacios en blanco.
   - Descarta textos de 3 caracteres o menos.
3. Ajustar los **sliders de calibración** (ver sección siguiente) según necesidad.
4. Presionar **🚀 Clasificar requerimientos con IA**.
5. Revisar la tabla de resultados y descargar el CSV clasificado con **📥 Descargar CSV Clasificado**.

## Sliders de calibración

| Slider | Qué controla | Efecto al subirlo | Efecto al bajarlo |
|---|---|---|---|
| **Umbral confianza (normales)** | Confianza mínima para aceptar descripciones largas | Más estricto: descarta más tickets ambiguos | Más permisivo: acepta tickets aunque el modelo dude |
| **Umbral confianza (cortas)** | Confianza mínima para descripciones cortas | Filtra frases cortas irrelevantes ("Hola") | Riesgo de dejar pasar frases cortas sin relación a IT |
| **Umbral de longitud (caracteres)** | Punto de corte que define qué es una descripción "corta" | Más textos caen en el régimen "normal" (más estricto) | Más textos caen en el régimen "corto" (más permiso) |

### Cómo calibrar en la práctica

1. Correr una primera clasificación con los valores por defecto.
2. Revisar la columna **`Confianza`** en la tabla de resultados.
3. Si se descartan tickets legítimos → bajar el umbral correspondiente (normal o corto según la longitud del texto descartado).
4. Si se cuelan textos irrelevantes → subir el umbral correspondiente.
5. Los sliders recalculan el filtro **instantáneamente** sin volver a correr el modelo — solo el botón "Clasificar" ejecuta la IA nuevamente.

## Columnas del resultado

| Columna | Descripción |
|---|---|
| `Area_Asignada` | Categoría asignada por el modelo (Hardware / Software / Redes) |
| `Confianza` | Score de 0 a 1 que indica cuán seguro está el modelo de esa asignación |
| `Longitud` | Cantidad de caracteres de la descripción, usada para decidir qué umbral aplicar |

## Notas técnicas

- El modelo procesa **zero-shot classification** con `multi_label=True`: cada categoría se evalúa de forma independiente, sin competir entre sí por sumar 1.
- Las categorías incluyen descripciones de contexto internas (no visibles en el resultado) para mejorar la precisión semántica en español.
- Todo el procesamiento es **local**: no se envían datos a servicios externos, apto para información sensible.

## Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| Error "El archivo debe contener una columna llamada 'descripcion'" | El CSV no tiene esa columna exacta | Renombrar la columna en el CSV antes de subir |
| Clasificación muy lenta | Corre en CPU sin GPU | Normal en CPU; considerar GPU para volúmenes grandes |
| Se descartan tickets válidos | Umbral demasiado alto | Bajar el umbral correspondiente (ver tabla de calibración) |
| Pasan textos irrelevantes | Umbral demasiado bajo | Subir el umbral correspondiente |
