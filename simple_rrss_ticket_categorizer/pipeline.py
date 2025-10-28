import re
import warnings

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling._random_over_sampler import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
nltk.download("wordnet", download_dir=".downloaded/")
nltk.download("stopwords", download_dir=".downloaded/")
nltk.data.path.append(".downloaded/")
lemmatizer = WordNetLemmatizer()


def plot_learning_curve(estimator, X, y, scoring="f1_weighted", cv=3):
    """
    Genera una curva de aprendizaje para diagnosticar overfitting/underfitting.
    'estimator' debe ser el modelo ya entrenado (ej. grid_search.best_estimator_)
    """
    print("\nGenerando Curvas de Aprendizaje...")

    # Define los tamaños del set de entrenamiento
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        # Genera 5 puntos en la curva
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=42,
    )

    # Calcula las medias y desviaciones
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.title(f"Curva de Aprendizaje (Scoring: {scoring})")
    plt.xlabel("Muestras de Entrenamiento")
    plt.ylabel("Puntuación (Score)")
    plt.grid(True)

    # Rellenar el área de desviación estándar
    plt.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes_abs,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )

    # Dibujar las líneas de media
    plt.plot(
        train_sizes_abs,
        train_scores_mean,
        "o-",
        color="r",
        label="Puntuación de Entrenamiento",
    )
    plt.plot(
        train_sizes_abs,
        test_scores_mean,
        "o-",
        color="g",
        label="Puntuación de Validación (Cross-Validation)",
    )

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("✅ Curva de Aprendizaje guardada en 'learning_curve.png'")


def plot_gridsearch_heatmap(grid_search, param1, param2):
    """
    Genera un heatmap de los resultados de GridSearchCV para dos parámetros.

    'grid_search' es el objeto grid_search *después* de .fit()
    'param1' y 'param2' son los nombres de los parámetros que quieres en los ejes.
    Ej: param1='classifier__C', param2='tfidf__max_features'
    """
    print("\nGenerando Heatmap de GridSearchCV...")

    # Convertir resultados a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)

    # Crear una tabla pivote
    try:
        pivot = results.pivot_table(
            values="mean_test_score",
            index=f"param_{param1}",
            columns=f"param_{param2}",
        )
    except Exception as e:
        print(f"Error al crear la tabla pivote para {param1} y {param2}: {e}")
        print(
            "Asegúrate de que los nombres de los parámetros coinciden con tu param_grid.",
        )
        return

    # Graficar el heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "F1-Score Ponderado (mean_test_score)"},
    )
    plt.title("Heatmap de Resultados de GridSearchCV")
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.tight_layout()
    plt.savefig("gridsearch_heatmap.png")
    print("✅ Heatmap de GridSearchCV guardado en 'gridsearch_heatmap.png'")


def clean_text_data(df):
    """Limpia y asegura que las columnas de texto no tengan NaN"""

    # Rellenar NaN con strings vacíos
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Asegurar que sean strings
    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)

    return df


def create_simple_categories(df: pd.DataFrame):
    """
    Crea categorías simplificadas basadas en el dataset,
    priorizando la 'queue' para mayor precisión.
    """
    print("=== CREANDO CATEGORÍAS SIMPLIFICADAS (Lógica Mejorada) ===")

    def assign_simple_category(row):
        ticket_type = str(row.get("type", "Unknown")).lower()
        queue = str(row.get("queue", "Unknown")).lower()

        # 1. Billing (La más específica)
        # Usamos 'in' para capturar 'billing and payments'
        if "billing" in queue or "payment" in queue:
            return "Billing_Support"

        # 2. Technical Support (Agrupa todos los tipos de soporte técnico)
        # Usamos 'type' Incident/Problem Y las colas técnicas
        if (
            ticket_type in ["incident", "problem"]
            or "technical support" in queue
            or "it support" in queue
            or "product support" in queue
            or "service outages" in queue
        ):
            return "Technical_Support"

        # 3. Service Management (Cambios, solicitudes de RH, etc.)
        if ticket_type == "change" or "human resources" in queue:
            return "Service_Management"

        # 4. Customer Service (El resto: Requests, General, Sales)
        # Todo lo que queda, que son principalmente consultas y solicitudes no técnicas.
        # (Incluye 'request', 'customer service', 'general inquiry', 'sales', 'returns')
        return "Customer_Service"

    df["Simple_Target"] = df.apply(assign_simple_category, axis=1)

    print("Distribución de categorías (Lógica Mejorada):")
    print(df["Simple_Target"].value_counts())

    return df


def add_text_features(df):
    """
    Agrega un conjunto robusto de features numéricas derivadas del texto
    para mejorar la clasificación.
    """

    # 0. Limpieza preliminar y creación de texto unificado
    # Asegura que no haya NaNs y crea un campo unificado para búsquedas de keywords
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["full_text"] = df["subject"] + " " + df["body"]

    # --- 1. Features de Longitud y Ratio ---
    df["subject_length"] = df["subject"].str.len()
    df["body_length"] = df["body"].str.len()
    df["subject_word_count"] = df["subject"].str.split().str.len()
    df["body_word_count"] = df["body"].str.split().str.len()

    # Feature: Ratio de mayúsculas en el asunto (señal de urgencia/enojo)
    df["subject_uppercase_ratio"] = df["subject"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1),
    )

    # Feature NUEVA: Ratio de mayúsculas en el cuerpo
    df["body_uppercase_ratio"] = df["body"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1),
    )

    # Feature NUEVA: Largo promedio de palabra (textos técnicos suelen tener palabras más largas)
    # Usamos np.where para evitar división por cero si body_word_count es 0
    df["avg_word_length_body"] = np.where(
        df["body_word_count"] > 0,
        df["body_length"] / df["body_word_count"],
        0,
    )

    # --- 2. Features de Puntuación y Caracteres Especiales ---

    # Busca en el texto completo (más eficiente)
    df["has_question_mark"] = (
        df["full_text"].str.contains(r"\?", regex=True).astype(int)
    )
    df["has_exclamation"] = df["full_text"].str.contains(r"!", regex=True).astype(int)

    # Feature NUEVA: Menciona algún símbolo de moneda (señal fuerte de Billing)
    df["mentions_currency"] = (
        df["full_text"].str.contains(r"\$|€|£|CLP|USD", regex=True).astype(int)
    )

    # Feature NUEVA: Contiene una mención (formato @usuario), sugiere red social
    df["is_social_media_mention"] = (
        df["full_text"].str.contains(r"\B@\w+", regex=True).astype(int)
    )

    # Feature NUEVA: Contiene dígitos (códigos de error, IDs, montos)
    df["has_digits"] = df["full_text"].str.contains(r"\d", regex=True).astype(int)

    # --- 3. Features de Palabras Clave (Keywords) ---
    # Definimos las listas de palabras clave (regex)

    # Technical_Support
    support_keywords = (
        r"\b(error|fallo|problema|caído|no funciona|lento|servidor|software|bug|down|broken|404|503|crash|"
        r"failure|outage|offline|disconnect|no conecta|sin acceso|no puedo entrar|exception|timeout)\b"
    )

    # Billing_Support (expandida)
    billing_keywords = (
        r"bill|billing|invoice|factura|boleta|cobro|pago|payment|charge|charged|"
        r"monto|amount|fee|cost|price|refund|rebate|overcharge|debit|credit|"
        r"saldo|deuda|abono|receipt|statement|account balance|duda"
    )

    # Urgencia (Genérica)
    urgent_keywords = (
        r"\b(urgente|crítico|inmediato|asap|ahora|inmediatamente|emergencia|priority|prioridad|"
        r"urgent|critical|immediate|now)\b"
    )
    # Customer_Service
    service_keywords = (
        r"\b(consulta|duda|info|información|portabilidad|planes|plan|contratar|ayuda|pregunta|horario|dirección|"
        r"question|query|help|information|assist|cómo|cuando|donde|what|how|where|when)\b"
    )
    # Feature NUEVA: Service_Management
    management_keywords = (
        r"\b(solicitud|cambio|configuración|actualizar|modificar|install|upgrade|baja|cancelar|cancellation|"
        r"request|change|configure|update|modify|installation|alta|activar|desactivar|activate|deactivate)\b"
    )
    # Feature NUEVA: Sentimiento Negativo (frustración)
    negative_sentiment_keywords = (
        r"\b(malo|pésimo|horrible|frustrado|molesto|rabia|basura|worst|terrible|queja|complaint|"
        r"decepcionado|enojado|harto|angry|frustrated|awful|sucks|bad|molestia|problemas)\b|"
        r"nunca funciona|siempre falla"  # Estos no necesitan \b
    )
    # Aplicamos las búsquedas sobre 'full_text' (mucho más rápido)
    df["mentions_error"] = (
        df["full_text"]
        .str.contains(support_keywords, case=False, regex=True)
        .astype(int)
    )
    df["mentions_billing"] = (
        df["full_text"]
        .str.contains(billing_keywords, case=False, regex=True)
        .astype(int)
    )
    df["mentions_urgent"] = (
        df["full_text"]
        .str.contains(urgent_keywords, case=False, regex=True)
        .astype(int)
    )
    df["mentions_service"] = (
        df["full_text"]
        .str.contains(service_keywords, case=False, regex=True)
        .astype(int)
    )
    df["mentions_management"] = (
        df["full_text"]
        .str.contains(management_keywords, case=False, regex=True)
        .astype(int)
    )
    df["mentions_sentiment_negative"] = (
        df["full_text"]
        .str.contains(negative_sentiment_keywords, case=False, regex=True)
        .astype(int)
    )

    # Ya no necesitamos la columna 'full_text', la podemos borrar
    df = df.drop(columns=["full_text"])

    return df


def build_simple_model():
    """Construye un modelo simple pero efectivo"""

    stopwords_list = [
        *stopwords.words("english"),
        *stopwords.words("french"),
        *stopwords.words("german"),
        # *stopwords.words("italian"),
        *stopwords.words("portuguese"),
        *stopwords.words("spanish"),
    ]

    subject_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    sublinear_tf=True,
                    norm="l2",
                    min_df=2,
                    max_df=0.85,
                    stop_words=stopwords_list,
                    strip_accents="unicode",
                    analyzer="word",
                    # dtype=np.float32,
                ),
            ),
        ],
    )

    body_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    sublinear_tf=True,
                    norm="l2",
                    min_df=4,
                    max_df=0.75,
                    stop_words=stopwords_list,
                    strip_accents="unicode",
                    analyzer="word",
                    dtype=np.float32,
                ),
            ),
        ],
    )

    numeric_features = [
        # Features de Longitud
        "subject_length",
        "body_length",
        "subject_word_count",
        "body_word_count",
        # Features de Ratio y Estructura
        "subject_uppercase_ratio",
        "body_uppercase_ratio",
        "avg_word_length_body",
        # Features de Puntuación y Caracteres
        "has_question_mark",
        "has_exclamation",
        "mentions_currency",
        "is_social_media_mention",
        "has_digits",
        # Features de Palabras Clave (Keywords)
        "mentions_error",
        "mentions_billing",
        "mentions_urgent",
        "mentions_service",
        "mentions_management",
        "mentions_sentiment_negative",
    ]

    feature_transformer = ColumnTransformer(
        [
            ("subject_features", subject_pipeline, "subject"),
            ("body_features", body_pipeline, "body"),
            ("numeric_features", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    base_svc = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
        dual=False,
        loss="squared_hinge",
        tol=1e-4,
    )

    model = ImbPipeline(
        [
            (
                "oversample",
                RandomOverSampler(random_state=42, sampling_strategy="not majority"),
            ),
            ("features", feature_transformer),  # El ColumnTransformer es el primer paso
            # ("svd", TruncatedSVD(n_components=300, random_state=42)),
            # (
            #    "oversample",
            #    SMOTE(random_state=42, k_neighbors=1, sampling_strategy="not majority"),
            # ),
            (
                "classifier",
                OneVsRestClassifier(base_svc, n_jobs=-1),
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
    tokens = [t for t in text.split() if len(t) > 2]
    # lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    # return " ".join(tokens)
    return " ".join(tokens)


def train_simple_model(df, target_col="Simple_Target"):
    print("\n=== ENTRENANDO MODELO SIMPLIFICADO CON GRIDSEARCHCV ===")

    numeric_features = [
        # Features de Longitud
        "subject_length",
        "body_length",
        "subject_word_count",
        "body_word_count",
        # Features de Ratio y Estructura
        "subject_uppercase_ratio",
        "body_uppercase_ratio",
        "avg_word_length_body",
        # Features de Puntuación y Caracteres
        "has_question_mark",
        "has_exclamation",
        "mentions_currency",
        "is_social_media_mention",
        "has_digits",
        # Features de Palabras Clave (Keywords)
        "mentions_error",
        "mentions_billing",
        "mentions_urgent",
        "mentions_service",
        "mentions_management",
        "mentions_sentiment_negative",
    ]
    text_features = ["subject", "body"]

    # X ahora contiene TODAS las columnas
    X = df[text_features + numeric_features]  # <--- CAMBIO AQUÍ
    y = df[target_col]
    # División estratificada
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Pipeline completo
    pipeline = build_simple_model()

    # Fase 1: Búsqueda amplia
    print("=== FASE 1: Búsqueda Amplia ===")
    phase1_param_grid = {
        "classifier__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "classifier__estimator__penalty": ["l2"],
        "features__subject_features__tfidf__max_features": [500, 1000, 2500],
        "features__subject_features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__body_features__tfidf__max_features": [1000, 2000, 4000, 6000],
        "features__body_features__tfidf__ngram_range": [(1, 2), (1, 3), (2, 3)],
    }

    phase1_search = HalvingGridSearchCV(
        pipeline,
        phase1_param_grid,
        cv=3,
        factor=2,
        scoring="f1_macro",
        n_jobs=8,
        verbose=10,
        random_state=42,
        error_score="raise",
    )

    phase1_search.fit(X_train, y_train)
    print(f"Mejores parámetros Fase 1: {phase1_search.best_params_}")

    # Fase 2: Búsqueda refinada alrededor de los mejores parámetros
    print("\n=== FASE 2: Búsqueda Refinada ===")
    best_phase1 = phase1_search.best_params_

    phase2_param_grid = {
        "classifier__estimator__C": np.linspace(
            max(0.01, best_phase1["classifier__estimator__C"] / 3),
            best_phase1["classifier__estimator__C"] * 3,
            5,
        ).tolist(),
        "features__subject_features__tfidf__max_features": [
            max(
                500,
                best_phase1["features__subject_features__tfidf__max_features"] - 1000,
            ),
            best_phase1["features__subject_features__tfidf__max_features"],
            min(
                8000,
                best_phase1["features__subject_features__tfidf__max_features"] + 1000,
            ),
        ],
        "features__body_features__tfidf__max_features": [
            max(
                3000,
                best_phase1["features__body_features__tfidf__max_features"] - 2000,
            ),
            best_phase1["features__body_features__tfidf__max_features"],
            min(
                25000,
                best_phase1["features__body_features__tfidf__max_features"] + 2000,
            ),
        ],
        "features__subject_features__tfidf__ngram_range": [(1, 2), (1, 3)],
        "features__body_features__tfidf__ngram_range": [(1, 2), (1, 3), (2, 3)],
    }

    # Usar el mejor estimador de la fase 1 como punto de partida
    phase2_search = HalvingGridSearchCV(
        phase1_search.best_estimator_,
        phase2_param_grid,
        cv=3,
        factor=2,
        scoring="f1_weighted",
        n_jobs=8,
        verbose=10,
        random_state=42,
    )

    phase2_search.fit(X_train, y_train)

    print(f"Mejores parámetros Fase 2: {phase2_search.best_params_}")

    model = phase2_search.best_estimator_
    # Curva de aprendizaje y heatmap (opcional)
    plot_learning_curve(model, X_train, y_train, scoring="f1_weighted")
    plot_gridsearch_heatmap(
        phase2_search,
        param1="classifier__estimator__C",
        param2="features__subject_features__tfidf__max_features",
    )

    # Evaluación
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Accuracy - Entrenamiento (Mejor Modelo): {train_score:.3f}")
    print(f"Accuracy - Prueba (Mejor Modelo): {test_score:.3f}")

    print("\n=== REPORTE DE CLASIFICACIÓN (Mejor Modelo) ===")
    print(classification_report(y_test, y_pred_test))

    return model, X_train, y_train


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

    data = add_text_features(data_to_predict)

    # 2. Predecir usando el DataFrame
    # El pipeline ahora encontrará las columnas 'subject' y 'body'
    prediction = model.predict(data)[0]

    print(f"\nAsunto: {subject}")
    print(f"Categoría predicha: {prediction}")

    return prediction


if __name__ == "__main__":
    # Cargar datos
    df_cleaned = pd.read_csv("data/processed/tickets_cleaned-3.csv")

    # 1. Limpiar datos de texto
    df_cleaned = clean_text_data(df_cleaned)

    df = add_text_features(df_cleaned)

    # 2. Crear categorías simplificadas
    df_final = create_simple_categories(df_cleaned)

    # 3. Entrenar modelo simple
    model, X_test, y_test = train_simple_model(df_final, target_col="Simple_Target")

    # 4. Guardar modelo
    joblib.dump(model, "models/simple_ticket_classifier.pkl")
    df_final.to_csv("data/processed/tickets_simple_categories.csv", index=False)

    print("\n MODELO SIMPLE GUARDADO")

    # 5. Probar con ejemplos
    test_cases = [
        "Problema crítico del servidor requiere atención inmediata",
        "Consulta sobre disponibilidad de producto",
        "Error en la facturación del servicio",
        "Solicitud de cambio de configuración",
        "Problema técnico con el software",
        "@[EmpresaX] me llegó la boleta y el monto es incorrecto, me están cobrando un plan que di de baja el mes pasado.",
        "@[EmpresaX] hola, quiero portarme a su compañía, ¿dónde veo los planes de fibra óptica??",
        "@[EmpresaX] otra vez caídos? qué servicio más malo... como siempre.",
    ]

    print("\n=== PRUEBAS RÁPIDAS ===")
    for test_case in test_cases:
        predict_ticket_simple(model, test_case)
