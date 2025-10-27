# data_cleaning.py

import pandas as pd


def load_data(filepath):
    """Carga los datos desde un archivo CSV o Excel."""
    if filepath.endswith(".csv"):
        df: pd.DataFrame = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        df: pd.DataFrame = pd.read_excel(filepath)
    else:
        raise ValueError("Formato de archivo no soportado. Usa .csv o .xlsx")
    if df is None:
        raise ValueError("El archivo no contiene datos válidos.")
    if df.empty:
        raise ValueError("El archivo está vacío o no contiene datos válidos.")
    return df


def clean_text_columns(df: pd.DataFrame):
    """Limpia columnas de texto: quita espacios, normaliza mayúsculas/minúsculas, elimina caracteres especiales innecesarios."""
    text_cols = ["subject", "body", "answer"]
    for col in text_cols:
        if col in df.columns:
            # Convertir a string y quitar espacios
            df[col] = df[col].astype(str).str.strip()
            # Eliminar múltiples espacios y reemplazar por uno solo
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
            # Quitar caracteres no alfanuméricos (excepto espacios y puntuación básica)
            df[col] = df[col].str.replace(r"[^\w\s.,!?;:()\-]", "", regex=True)
    return df


def handle_missing_values(df: pd.DataFrame):
    """Rellena o elimina valores faltantes según la columna."""
    # Para columnas críticas: type, queue, priority
    df["type"] = df["type"].fillna("Unknown")
    df["queue"] = df["queue"].fillna("Unknown")
    df["priority"] = df["priority"].fillna("medium")

    # Para language: si está vacío, asignar "en" (inglés) como default
    df["language"] = df["language"].fillna("en")

    # Para business_type: si está vacío, asignar "Unknown"
    df["business_type"] = df["business_type"].fillna("Unknown")

    # Para subject y body: llenar con texto por defecto si están vacíos
    df["subject"] = df["subject"].fillna("No subject provided")
    df["body"] = df["body"].fillna("No content provided")
    df["answer"] = df["answer"].fillna("No resolution provided")

    # Para las columnas de tags (tag_1 a tag_9): llenar con string vacío
    tag_columns = [f"tag_{i}" for i in range(1, 10)]
    for tag_col in tag_columns:
        if tag_col in df.columns:
            df[tag_col] = df[tag_col].fillna("")

    return df


def extract_features_from_description(df: pd.DataFrame):
    """Extrae características útiles del texto de la descripción."""
    # Combinar subject y body para análisis de texto
    df["combined_text"] = df["subject"] + " " + df["body"]

    # Contar longitud del texto
    df["text_length"] = df["combined_text"].str.len()
    df["word_count"] = df["combined_text"].str.split().str.len()

    # Detectar palabras clave relacionadas con urgencia
    urgency_keywords = [
        "urgent",
        "critical",
        "immediate",
        "asap",
        "emergency",
        "urgente",
        "crítico",
        "inmediato",
        "emergencia",
        "dringend",
        "kritisch",
        "sofortig",
        "urgence",
        "critique",
        "immédiat",
    ]

    df["has_urgency_keyword"] = (
        df["combined_text"]
        .str.contains(
            "|".join(urgency_keywords),
            case=False,
            na=False,
        )
        .astype(int)
    )

    # Detectar si hay errores técnicos
    error_keywords = [
        "error",
        "bug",
        "fail",
        "crash",
        "broken",
        "issue",
        "problem",
        "error",
        "falla",
        "trabado",
        "problema",
        "incidente",
        "fehler",
        "absturz",
        "kaputt",
        "problem",
        "erreur",
        "bug",
        "panne",
        "casse",
        "problème",
    ]

    df["has_error_keyword"] = (
        df["combined_text"]
        .str.contains(
            "|".join(error_keywords),
            case=False,
            na=False,
        )
        .astype(int)
    )

    return df


def create_target_column(df: pd.DataFrame):
    """Crea columnas objetivo para clasificación basadas en el dataset."""
    # Opción 1: Usar 'type' como objetivo principal
    df["target_type"] = df["type"].copy()

    # Opción 2: Usar 'queue' como objetivo secundario
    df["target_queue"] = df["queue"].copy()

    # Opción 3: Crear categoría combinada para clasificación más granular
    df["target_combined"] = df["type"] + "_" + df["queue"]

    return df


def encode_categorical_variables(df: pd.DataFrame):
    """Codifica variables categóricas para modelado."""
    # Columnas categóricas principales
    categorical_columns = ["type", "queue", "priority", "language", "business_type"]

    # Crear dummies para las categorías principales
    for col in categorical_columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

    # Crear características de tags (combinar todos los tags en una lista)
    tag_columns = [f"tag_{i}" for i in range(1, 10) if f"tag_{i}" in df.columns]
    if tag_columns:
        df["all_tags"] = df[tag_columns].apply(
            lambda x: ",".join(x.dropna().astype(str)),
            axis=1,
        )
        # Crear dummies para los tags más comunes
        all_tags_combined = df[tag_columns].stack().value_counts()
        top_tags = all_tags_combined.head(20).index  # Top 20 tags más comunes

        for tag in top_tags:
            df[f"tag_{tag.replace(' ', '_').lower()}"] = df[tag_columns].apply(
                lambda x: 1 if tag in x.values else 0,
                axis=1,
            )

    return df


def preprocess_data(df: pd.DataFrame):
    """Función principal que aplica todas las transformaciones."""
    print("Iniciando limpieza de datos...")
    df = clean_text_columns(df)
    df = handle_missing_values(df)
    df = extract_features_from_description(df)
    df = create_target_column(df)
    df = encode_categorical_variables(df)
    print("Limpieza completada.")
    return df


if __name__ == "__main__":
    # Ejemplo de uso
    df = load_data(
        "data/raw/dataset-tickets-multi-lang3-4k.csv",
    )  # Actualiza con la ruta correcta
    df_cleaned = preprocess_data(df)
    df_cleaned.to_csv("data/processed/tickets_cleaned-3.csv", index=False)
    print("Datos guardados en 'tickets_cleaned-3.csv'")
    print(f"Shape del dataset limpio: {df_cleaned.shape}")
    print(f"Columnas disponibles: {df_cleaned.columns.tolist()}")
