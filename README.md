# Clasificador de Tickets de Soporte para Redes Sociales

Este proyecto implementa un modelo de Machine Learning para clasificar automáticamente tickets de soporte (como _tweets_ o correos) en categorías de negocio predefinidas. La solución está diseñada para optimizar los procesos de atención al cliente, reduciendo los tiempos de respuesta y mejorando la eficiencia operativa.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)](https://scikit-learn.org/stable/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.x-green)](https://fastapi.tiangolo.com/)

---

## Índice

- [Resumen del Proyecto](#resumen-del-proyecto)
- [Problema de Negocio](#problema-de-negocio)
- [Solución Propuesta](#solución-propuesta)
- [Características Principales](#características-principales)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
  - [1. Preprocesamiento de Datos](#1-preprocesamiento-de-datos)
  - [2. Entrenamiento del Modelo](#2-entrenamiento-del-modelo)
  - [3. Generación de Gráficos de Evaluación](#3-generación-de-gráficos-de-evaluación)
  - [4. Ejecución del Servidor de Demostración (API)](#4-ejecución-del-servidor-de-demostración-api)
- [Mejoras Futuras](#mejoras-futuras)

---

## Resumen del Proyecto

Este trabajo presenta una solución de Inteligencia Artificial para optimizar la atención al cliente en redes sociales. Ante el alto volumen de menciones, los equipos de soporte sufren de altos tiempos de respuesta (FRT) al tener que clasificar manualmente la intención de cada mensaje.

Se implementó un modelo de Machine Learning clásico (**Support Vector Machine**) en Python con **Scikit-learn** para clasificar automáticamente la intención del cliente (ej. "Queja Técnica", "Consulta Facturación"). Esta solución permite a las empresas reducir el tiempo de triaje de horas a segundos, impactando directamente los KPIs de satisfacción del cliente.

## Problema de Negocio

La gestión de interacciones en redes sociales es un cuello de botella para muchas empresas B2C. El proceso manual de **triaje y enrutamiento** de tickets es:
- **Ineficiente y costoso:** Consume horas-hombre en tareas repetitivas.
- **Deteriora la experiencia del cliente:** Aumenta los tiempos de respuesta (FRT) y resolución (ART).
- **Genera pérdida de oportunidades:** Las intenciones de compra o las crisis reputacionales se atienden con lentitud.

## Solución Propuesta

Se desarrolló un pipeline de Procesamiento de Lenguaje Natural (NLP) que:
1.  **Preprocesa y limpia** el texto de los tickets (tweets, correos).
2.  **Extrae características relevantes** del texto mediante ingeniería de features (conteo de palabras, mención de keywords, etc.) y vectorización **TF-IDF**.
3.  **Clasifica el ticket** utilizando un clasificador **Linear Support Vector Machine (LinearSVC)**, optimizado para manejar texto de alta dimensionalidad.
4.  **Expone el modelo** a través de una **API REST** para una fácil integración con sistemas CRM existentes (Zendesk, Salesforce, etc.).

## Características Principales

-   **Preprocesamiento Avanzado:** Limpieza de texto, manejo de múltiples idiomas y creación de una columna `category` unificada a partir de reglas de negocio.
-   **Ingeniería de Features:** Creación de más de 20 features numéricas a partir del texto para enriquecer el modelo (ej. `mentions_billing`, `has_question_mark`, `subject_uppercase_ratio`).
-   **Pipeline de ML Robusto:** Uso de `ColumnTransformer` para procesar features de texto y numéricas de forma independiente.
-   **Optimización de Hiperparámetros:** Búsqueda automática de los mejores parámetros del modelo usando `HalvingGridSearchCV`.
-   **Manejo de Desbalance de Clases:** Implementación de `RandomOverSampler` para mejorar el rendimiento en categorías minoritarias.
-   **Evaluación y Visualización:** Scripts para generar matrices de confusión, reportes de clasificación, curvas de aprendizaje, curvas ROC y Precision-Recall.
-   **API de Inferencia:** Servidor web con FastAPI para realizar predicciones en tiempo real.

## Stack Tecnológico

-   **Lenguaje:** Python 3.9+
-   **Análisis de Datos:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, Imbalanced-learn
-   **NLP:** NLTK
-   **API:** FastAPI, Uvicorn
-   **Visualización:** Matplotlib, Seaborn
-   **Serialización de Modelos:** Joblib


## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/DiegoRiveraEstefano/simple-rrss-ticket-categorizer
    cd simple-rrss-ticket-categorizer
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar recursos de NLTK:**
    ```bash
    python -m nltk.downloader wordnet omw-1.4
    ```

## Uso

Sigue estos pasos en orden para replicar el flujo de trabajo completo.

### 1. Preprocesamiento de Datos

Este script carga el dataset crudo, lo limpia, extrae features numéricas, crea la columna objetivo `category` y guarda el resultado en `data/processed/`.

-   **Requisito:** Asegúrate de tener tu dataset crudo en `data/raw/` (ej. `dataset-tickets-multi-lang3-4k.csv`).
-   **Ejecución:**
    ```bash
    python -m simple_rrss_ticket_categorizer.preprocessing
    ```
-   **Salida:** Se creará un archivo `tickets_cleaned_con_category.csv` en la carpeta `data/processed/`.

### 2. Entrenamiento del Modelo

Este script toma los datos preprocesados, entrena el modelo `LinearSVC` buscando los mejores hiperparámetros y guarda el modelo final.

-   **Requisito:** Haber ejecutado el paso de preprocesamiento.
-   **Ejecución:**
    ```bash
    python -m simple_rrss_ticket_categorizer.pipeline
    ```
-   **Salida:**
    -   El modelo entrenado se guardará como `simple_ticket_classifier.pkl` en la carpeta `models/`.
    -   Se generarán los gráficos `learning_curve.png` y `gridsearch_heatmap.png` en la raíz del proyecto.
    -   Se mostrará en consola el reporte de clasificación del modelo.

### 3. Generación de Gráficos de Evaluación

Este script carga el modelo guardado y el conjunto de datos de prueba para generar visualizaciones detalladas del rendimiento.

-   **Requisito:** Haber entrenado y guardado un modelo en `models/`.
-   **Ejecución:**
    ```bash
    python -m simple_rrss_ticket_categorizer.diagrams
    ```
-   **Salida:** Se generarán los siguientes archivos en la raíz del proyecto:
    -   `class_distribution.png`
    -   `confusion_matrix_counts.png`
    -   `confusion_matrix_normalized.png`
    -   `classification_report.png`
    -   `precision_recall_curves.png`
    -   `roc_auc_curves.png`

### 4. Ejecución del Servidor de Demostración (API)

Inicia un servidor web local con FastAPI que expone el modelo para hacer predicciones en tiempo real.

-   **Requisito:** Haber entrenado y guardado un modelo en `models/`.
-   **Ejecución:**
    ```bash
    python -m app.main
    ```
-   **Acceso:**
    -   **API Docs (Swagger UI):** Abre tu navegador y ve a http://127.0.0.1:8000/docs para interactuar con la API.
    -   **Demo Frontend:** Abre http://127.0.0.1:8000/ para ver una interfaz de demostración simple.




