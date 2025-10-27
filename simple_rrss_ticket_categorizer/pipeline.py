import re
import warnings

import joblib
import nltk
import pandas as pd

# from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection._search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
nltk.download("wordnet", download_dir=".downloaded/")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


def clean_text_data(df):
    """Limpia y asegura que las columnas de texto no tengan NaN"""

    # Rellenar NaN con strings vacíos
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Asegurar que sean strings
    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)

    return df


def create_simple_categories(df):
    """Crea categorías simplificadas basadas en el dataset"""

    print("=== CREANDO CATEGORÍAS SIMPLIFICADAS ===")

    # Mapeo directo basado en las columnas existentes
    def assign_simple_category(row):
        # Usar la columna 'type' como base y refinar con 'queue'
        ticket_type = row.get("type", "Unknown")
        queue = str(row.get("queue", "Unknown")).lower()

        if ticket_type == "Incident":
            return "Technical_Support"
        if ticket_type == "Request":
            if "billing" in queue or "payment" in queue:
                return "Billing_Support"
            return "Customer_Service"
        if ticket_type == "Problem":
            return "Technical_Support"
        if ticket_type == "Change":
            return "Service_Management"
        return "General_Support"

    df["Simple_Target"] = df.apply(assign_simple_category, axis=1)

    print("Distribución de categorías:")
    print(df["Simple_Target"].value_counts())

    return df


def build_simple_model():
    """Construye un modelo simple pero efectivo"""

    nltk.download("stopwords", download_dir=".downloaded/")
    stopwords_list = [
        *stopwords.words("english"),
        *stopwords.words("french"),
        *stopwords.words("german"),
        *stopwords.words("italian"),
        *stopwords.words("portuguese"),
        *stopwords.words("spanish"),
    ]

    text_processing_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    # ¡LA MAGIA ESTÁ AQUÍ!
                    # TfidfVectorizer aplicará esta función a cada documento
                    preprocessor=preprocess_text,
                    # El resto de tus parámetros (los valores base)
                    sublinear_tf=True,
                    norm="l2",
                    min_df=3,
                    max_df=0.85,
                    stop_words=stopwords_list,
                    strip_accents="unicode",
                    # max_features y ngram_range serán establecidos por GridSearchCV
                ),
            ),
        ],
    )

    feature_transformer = ColumnTransformer(
        [
            ("subject_features", text_processing_pipeline, "subject"),
            ("body_features", text_processing_pipeline, "body"),
        ],
        remainder="drop",  # No usar otras columnas
    )

    model = Pipeline(
        [
            ("features", feature_transformer),  # El ColumnTransformer es el primer paso
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=10000,
                    dual=False,
                    loss="squared_hinge",
                ),
            ),
        ],
    )

    return model


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|mailto:\S+", "", text)  # URLs/correos
    text = re.sub(r"\d+", "", text)  # números (opcional)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", " ", text)
    ##tokens = text.split()
    # stemmed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # return " ".join(stemmed_tokens)
    return text


def train_simple_model(df, target_col="Simple_Target"):
    """Entrena el modelo simplificado usando GridSearchCV"""

    print("\n=== ENTRENANDO MODELO SIMPLIFICADO CON GRIDSEARCHCV ===")

    X = df[["subject", "body"]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 1. Construir el pipeline (asegúrate que sea el de la Solución 1)
    pipeline = build_simple_model()

    # 2. Definir la grilla de parámetros a probar
    # 'tfidf__C' accede al parámetro 'C' del paso 'classifier'
    param_grid = {
        # Parámetros del clasificador
        "classifier__C": [0.01, 0.1, 0.5],
        "classifier__penalty": ["l1", "l2"],
        # Parámetros para el TF-IDF del ASUNTO
        "features__subject_features__tfidf__max_features": [1000, 3000, 5000],
        "features__subject_features__tfidf__ngram_range": [(1, 2), (1, 3)],
        # Parámetros para el TF-IDF del CUERPO
        "features__body_features__tfidf__max_features": [3000, 5000, 10000],
        "features__body_features__tfidf__ngram_range": [(1, 2)],
    }

    # 3. Configurar GridSearchCV
    # Buscará el mejor 'f1_weighted' para manejar desbalanceo
    # cv=3 (3-fold cross-validation) es más rápido que el default de 5
    # n_jobs=-1 usa todos tus núcleos de CPU
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )

    print("Iniciando búsqueda de hiperparámetros...")
    grid_search.fit(X_train, y_train)

    # 4. Usar el mejor modelo encontrado
    print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
    model = grid_search.best_estimator_

    # 5. Evaluación
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Accuracy - Entrenamiento (Mejor Modelo): {train_score:.3f}")
    print(f"Accuracy - Prueba (Mejor Modelo): {test_score:.3f}")

    print("\n=== REPORTE DE CLASIFICACIÓN (Mejor Modelo) ===")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test


def predict_ticket_simple(model, subject, body=""):
    """Predice la categoría de forma simple"""

    # Asegurar que los inputs sean strings
    subject = str(subject) if subject is not None else ""
    body = str(body) if body is not None else ""

    # 1. Crear un DataFrame con las columnas correctas
    # (exactamente como los datos de entrenamiento)
    data_to_predict = pd.DataFrame(
        {
            "subject": [subject],  # El dato debe estar dentro de una lista
            "body": [body],  # El dato debe estar dentro de una lista
        },
    )

    # 2. Predecir usando el DataFrame
    # El pipeline ahora encontrará las columnas 'subject' y 'body'
    prediction = model.predict(data_to_predict)[0]

    print(f"\nAsunto: {subject}")
    print(f"Categoría predicha: {prediction}")

    return prediction


if __name__ == "__main__":
    # Cargar datos
    df_cleaned = pd.read_csv("data/processed/tickets_cleaned-3.csv")

    # 1. Limpiar datos de texto
    df_cleaned = clean_text_data(df_cleaned)

    # 2. Crear categorías simplificadas
    df_final = create_simple_categories(df_cleaned)

    # 3. Entrenar modelo simple
    model, X_test, y_test = train_simple_model(df_final)

    # 4. Guardar modelo
    joblib.dump(model, "models/simple_ticket_classifier.pkl")
    df_final.to_csv("data/processed/tickets_simple_categories.csv", index=False)

    print("\n✅ MODELO SIMPLE GUARDADO")

    # 5. Probar con ejemplos
    test_cases = [
        "Problema crítico del servidor requiere atención inmediata",
        "Consulta sobre disponibilidad de producto",
        "Error en la facturación del servicio",
        "Solicitud de cambio de configuración",
        "Problema técnico con el software",
    ]

    print("\n=== PRUEBAS RÁPIDAS ===")
    for test_case in test_cases:
        predict_ticket_simple(model, test_case)
