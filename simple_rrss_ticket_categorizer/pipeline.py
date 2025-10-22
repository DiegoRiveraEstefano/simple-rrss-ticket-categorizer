from logging import getLogger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import create_preprocessor
from .preprocessing import feature_engineer_datetimes

# Importa las funciones de preprocesamiento del otro archivo
from .preprocessing import load_data

# --- Constantes del Proyecto ---
# Define las carpetas y nombres de archivos
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FILE_NAME = "it_support_tickets.csv"
TRAIN_DATA_FILE = RAW_DATA / FILE_NAME  # Asume que este es tu CSV
MODEL_NAME = "it_ticket_regressor.joblib"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# Define la variable objetivo y las columnas a dropear
TARGET_VARIABLE = "Resolution_Time_Hours"

logger = getLogger(__name__)


def create_pipeline() -> Pipeline:
    """
    Crea el pipeline completo de scikit-learn uniendo el
    preprocesador y el modelo de regresión.
    """

    # 1. Obtiene el preprocesador de columnas
    preprocessor = create_preprocessor()

    # 2. Define el modelo de Machine Learning
    # Usamos RandomForestRegressor, un modelo robusto que funciona bien
    # sin mucho ajuste de hiperparámetros.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # 3. Crea el Pipeline final
    # Este pipeline ejecutará primero todo el preprocesamiento
    # y luego pasará los datos limpios al RandomForest.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ],
    )


def load_data_raw_or_processed(raw_data: Path) -> tuple[pd.DataFrame, bool]:
    """
    Carga los datos de entrenamiento o prueba.
    Si no existe el archivo de entrenamiento, se carga el archivo de prueba.
    Retorna un DataFrame y un booleano indicando si los datos son procesados.
    """
    if (PROCESSED_DATA / FILE_NAME).exists():
        return load_data(str(PROCESSED_DATA / FILE_NAME)), True
    return load_data(str(RAW_DATA / FILE_NAME)), False


def train_pipeline():
    """
    Función principal para entrenar y guardar el pipeline.
    """
    logger.info("Iniciando el proceso de entrenamiento...")

    # Asegura que la carpeta de modelos exista
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    logger.info(f"Cargando datos desde {TRAIN_DATA_FILE}...")  # noqa: G004
    df, is_processed = load_data_raw_or_processed(TRAIN_DATA_FILE)

    if df.empty:
        logger.warning("No se pudieron cargar los datos. Abortando.")
        return

    if not is_processed:
        # 2. Ingeniería de Características (Fechas)
        logger.info("Realizando ingeniería de características (datetime)...")
        df = feature_engineer_datetimes(df)

        # 3. Eliminar filas donde el objetivo es nulo
        df = df.dropna(subset=[TARGET_VARIABLE])

        # 4. Guardar datos procesados para futuros usos
        logger.info(f"Guardando datos procesados en {PROCESSED_DATA / FILE_NAME}...")  # noqa: G004
        PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA / FILE_NAME, index=False)

    # 4. Definir X (features) e y (target)
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])  # El preprocesador se encarga del resto

    # 5. Dividir en Train/Test
    logger.info("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 6. Crear y Entrenar el Pipeline
    logger.info("Creando pipeline...")
    pipeline = create_pipeline()

    logger.info("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    # 7. Evaluar el modelo
    logger.info("Evaluando modelo...")
    y_pred = pipeline.predict(X_test)

    # Usamos RMSE (Root Mean Squared Error) para regresión
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"Evaluación completada. RMSE en Test: {rmse:.4f} horas")

    # 8. Guardar el pipeline entrenado
    logger.info(f"Guardando modelo en {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)

    logger.info("--- Proceso de entrenamiento finalizado ---")


if __name__ == "__main__":
    # Esto permite ejecutar el archivo directamente desde la terminal
    # para entrenar el modelo.
    # python -m simple_rrss_ticket_categorizer.pipeline
    train_pipeline()
