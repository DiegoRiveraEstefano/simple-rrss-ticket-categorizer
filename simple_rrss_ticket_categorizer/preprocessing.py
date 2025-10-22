import logging as logger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

logger = logger.getLogger(__name__)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carga y limpia inicialmente el dataset de tickets de IT.

    Maneja los formatos de fecha y los decimales con coma.
    """
    try:
        # El ejemplo muestra comas como decimales
        df = pd.read_csv(
            csv_path,
            decimal=",",  # Clave para '2,58'
            parse_dates=["Created_Date", "Resolved_Date"],  # Intenta parsear fechas
            dayfirst=True,  # Asume formato DD/MM/YYYY
        )
    except Exception as e:
        logger.exception(f"Error cargando el CSV: {e}")  # noqa: G004, TRY401
        return pd.DataFrame()

    # Asegura que las columnas de 'Yes/No' sean string para el preprocesador
    df["Automation_Used"] = df["Automation_Used"].astype(str)
    df["Resolved_First_Contact"] = df["Resolved_First_Contact"].astype(str)

    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Crea un ColumnTransformer de Scikit-learn para procesar los datos
    de los tickets de IT.
    """

    # --- 1. Definir listas de columnas por tipo ---

    # Columnas numéricas que necesitan imputación (mediana) y escalado
    numeric_features = ["Customer_Satisfaction"]

    # Columnas categóricas que necesitan imputación (moda) y One-Hot Encoding
    categorical_features = ["Ticket_Type", "Channel"]

    # Columnas binarias (Yes/No) que necesitan imputación y codificación ordinal
    # Se especifica el orden para que 'No' sea 0 y 'Yes' sea 1
    binary_features = ["Automation_Used", "Resolved_First_Contact"]
    binary_categories = [["No", "Yes"], ["No", "Yes"]]

    # Columnas de Datetime para ingeniería de características
    datetime_features = ["Created_Date"]

    # Columnas a eliminar
    # Ticket_ID no tiene valor predictivo.
    # Resolved_Date causa 'fuga de datos' si predecimos Resolution_Time.
    drop_features = ["Ticket_ID", "Resolved_Date"]

    # --- 2. Crear los pipelines de transformación ---

    # Pipeline para features numéricas
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )

    # Pipeline para features categóricas
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ],
    )

    # Pipeline para features binarias
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Rellena 'nan'
            ("ordinal", OrdinalEncoder(categories=binary_categories)),
        ],
    )

    # Pipeline para features de fecha (Ingeniería de Características)
    # Esta es una forma de extraer features de fechas dentro de un pipeline
    class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
        def fit(self, x, y=None):
            return self

        def transform(self, x, y=None):
            df = pd.DataFrame(x, columns=np.array(datetime_features))
            # Asegura que sea datetime
            df["Created_Date"] = pd.to_datetime(df["Created_Date"])

            features = pd.DataFrame()
            features["hour_created"] = df["Created_Date"].dt.hour
            features["dayofweek_created"] = df["Created_Date"].dt.dayofweek
            features["month_created"] = df["Created_Date"].dt.month
            return features

    # NOTA: Usar clases custom puede ser complejo. Un enfoque más simple
    # es hacer la ingeniería de fechas en Pandas ANTES de llamar al preprocesador.

    # --- 3. Crear el ColumnTransformer (Enfoque Simplificado) ---
    # Este preprocesador asume que la ingeniería de fechas (hour, dayofweek)
    # se hizo ANTES de pasar el DataFrame.

    # Nuevas listas de columnas (asumiendo que se corre una función previa)
    # numeric_features_expanded = ['Customer_Satisfaction', 'hour_created',
    #                              'dayofweek_created', 'month_created']

    # categorical_features = ['Ticket_Type', 'Channel']  # noqa: ERA001
    # binary_features = ['Automation_Used', 'Resolved_First_Contact']  # noqa: ERA001

    # --- Enfoque Final y Robusto (sin clases custom) ---
    # Es más limpio hacer la ing. de fechas fuera del pipeline.
    # Este preprocesador se aplicará a un DF que ya tiene esas columnas.

    numeric_features = [
        "Customer_Satisfaction",
        "hour_created",
        "dayofweek_created",
        "month_created",
    ]

    categorical_features = ["Ticket_Type", "Channel"]

    binary_features = ["Automation_Used", "Resolved_First_Contact"]

    # El preprocesador final
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
            # Se especifica 'drop' para las columnas que no listamos
            ("drop_cols", "drop", ["Ticket_ID", "Created_Date", "Resolved_Date"]),
        ],
        remainder="passthrough",  # Mantiene columnas no especificadas (como la
        # target si se dejó)
    )


def feature_engineer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características (features) a partir de las columnas de fecha.
    """
    df_out = df.copy()
    # Asegura que la columna es datetime
    df_out["Created_Date"] = pd.to_datetime(df_out["Created_Date"])

    df_out["hour_created"] = df_out["Created_Date"].dt.hour
    df_out["dayofweek_created"] = df_out["Created_Date"].dt.dayofweek
    df_out["month_created"] = df_out["Created_Date"].dt.month

    return df_out


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extrae los nombres de las features del ColumnTransformer."""
    feature_names = []

    # Nombres de features numéricas y de fechas
    feature_names.extend(preprocessor.transformers_[0][2])

    # Nombres de features categóricas (OneHotEncoded)
    onehot_cols = preprocessor.named_transformers_["cat"][
        "onehot"
    ].get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names.extend(onehot_cols)

    # Nombres de features binarias
    feature_names.extend(preprocessor.transformers_[2][2])

    return feature_names
