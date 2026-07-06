🎫 Sistema Inteligente de Triaje de Soporte (Pandas + LLM)

Este proyecto es una aplicación web interactiva diseñada para automatizar la gestión, limpieza y clasificación de requerimientos o tickets de soporte técnico. El sistema permite consolidar un flujo de trabajo de punta a punta, procesando datos tabulares crudos y enriqueciéndolos mediante modelos avanzados de Inteligencia Artificial.

📌 Función del Sistema

La aplicación cumple el rol de un **Asistente de Triaje Automatizado**. Su objetivo principal es recibir un volumen de datos crudos (en formato CSV) ingresados por un usuario, aplicarles un riguroso proceso de preprocesamiento estadístico para eliminar ruidos o redundancias, y posteriormente utilizar un modelo de lenguaje (LLM) para categorizar la naturaleza de cada ticket en una de tres áreas de resolución: **Hardware**, **Software** o **Redes**.

Clases del Curso Aplicadas e Integradas:
**Clase 2 (Eje Pandas - Limpieza de Datos): Carga dinámica de datasets considerando excepciones de codificación (`utf-8` / `latin1`). Implementa la remoción de registros nulos mediante `dropna()`, la eliminación de duplicados redundantes con `drop_duplicates()` y la normalización de cadenas de texto empleando transformaciones de tipo y remoción de espacios con `str.strip()`.
**Clase 6 (Eje LLMs - Inferencia Local): Integración directa del pipeline de Hugging Face (`zero-shot-classification`) usando el modelo de red neuronal `facebook/bart-large-mnli`. Realiza la inferencia local sobre el DataFrame limpio por Pandas, prediciendo la categoría más probable sin requerir peticiones a APIs externas propensas a bloqueos de red.

🚀 Guía de Uso Paso a Paso

Para interactuar con el sistema, seguí estos pasos:

1. Preparación del Archivo de Entrada (CSV)
El sistema requiere un archivo con extensión `.csv` que contenga una columna obligatoria llamada **`descripcion`** (en minúsculas), donde se detallen los problemas. Podés usar el siguiente formato de prueba:

```csv
id,descripcion
1,La impresora de secretaría no toma las hojas y hace un ruido extraño al encender.
2,La impresora de secretaría no toma las hojas y hace un ruido extraño al encender.
3,
4,Necesito que me reinstalen la licencia de Excel porque expiró esta mañana.
5,El módem principal parpadea en luz naranja y nos quedamos sin conexión wifi.
