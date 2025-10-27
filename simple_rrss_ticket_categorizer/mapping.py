import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def deep_data_investigation(df):
    """Investigación profunda de la calidad de los datos"""

    print("=== INVESTIGACIÓN PROFUNDA DE DATOS ===")

    # 1. Verificar si el texto es único o repetitivo
    print("\n1. ANÁLISIS DE UNICIDAD DE TEXTO:")
    text_duplicates = df.duplicated(
        subset=["Ticket Subject", "Ticket Description"]
    ).sum()
    print(f"Tickets duplicados exactos: {text_duplicates}")

    # Verificar similitud semántica
    df["text_length"] = df["Ticket Subject"] + " " + df["Ticket Description"]
    df["text_length"] = df["text_length"].str.len()

    # 2. Análisis de patrones de texto
    print("\n2. PATRONES DE TEXTO COMUNES:")
    sample_texts = df["Ticket Subject"].str.lower().value_counts().head(10)
    print("Asuntos más comunes:")
    for text, count in sample_texts.items():
        print(f"  '{text}': {count} ocurrencias")

    # 3. Análisis de bigramas más comunes
    print("\n3. BIGRAMAS MÁS COMUNES:")
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 2), max_features=20, stop_words="english"
    )
    X = vectorizer.fit_transform(df["Ticket Subject"] + " " + df["Ticket Description"])
    feature_names = vectorizer.get_feature_names_out()
    print("Bigramas más frecuentes:")
    for feature in feature_names:
        print(f"  {feature}")

    return df


def check_label_consistency(df, target_col="Target"):
    """Verifica la consistencia de las etiquetas"""

    print("\n=== CONSISTENCIA DE ETIQUETAS ===")

    # Agrupar por texto similar y ver distribución de etiquetas
    from collections import defaultdict

    # Buscar patrones comunes en el texto
    text_patterns = defaultdict(list)

    for idx, row in df.iterrows():
        # Extraer palabras clave del asunto
        subject_words = set(
            row["Ticket Subject"].lower().split()[:3]
        )  # Primeras 3 palabras
        pattern_key = " ".join(sorted(subject_words))
        text_patterns[pattern_key].append(row[target_col])

    print("Patrones de texto y distribución de etiquetas:")
    for pattern, labels in list(text_patterns.items())[:10]:
        if len(labels) > 2:  # Solo patrones con múltiples ejemplos
            label_counts = pd.Series(labels).value_counts()
            print(f"\nPatrón: '{pattern}'")
            print(f"  Distribución: {dict(label_counts)}")
            if len(label_counts) > 1:
                print(f"  ⚠️  INCONSISTENTE: {len(label_counts)} etiquetas diferentes")


def create_simplified_categories(df):
    """Crea categorías simplificadas basadas en análisis de texto"""

    print("\n=== CREANDO CATEGORÍAS SIMPLIFICADAS ===")

    # Mapeo basado en palabras clave (ajusta según tu dominio)
    category_keywords = {
        "Technical": [
            "issue",
            "problem",
            "error",
            "bug",
            "technical",
            "setup",
            "install",
            "network",
        ],
        "Financial": [
            "billing",
            "refund",
            "payment",
            "charge",
            "price",
            "cost",
            "money",
        ],
        "Product": [
            "product",
            "feature",
            "how to",
            "use",
            "functionality",
            "capability",
        ],
        "Service": [
            "cancel",
            "terminate",
            "service",
            "account",
            "access",
            "subscription",
        ],
    }

    def assign_simplified_category(text):
        text_lower = text.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return "Other"

    df["Simple_Target"] = df["Ticket Subject"].apply(assign_simplified_category)

    print("Distribución de categorías simplificadas:")
    print(df["Simple_Target"].value_counts())

    return df


def train_simplified_model(df, target_col="Simple_Target"):
    """Entrena modelo con categorías simplificadas"""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    print(f"\n=== ENTRENANDO MODELO SIMPLIFICADO ({target_col}) ===")

    # Usar solo texto para empezar
    X = df["Ticket Subject"] + " " + df["Ticket Description"]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            ("svm", LinearSVC(C=0.5, class_weight="balanced", random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Accuracy - Entrenamiento: {train_score:.3f}")
    print(f"Accuracy - Prueba: {test_score:.3f}")

    y_pred = model.predict(X_test)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    return model


def manual_data_validation(df):
    """Validación manual de una muestra de datos"""

    print("\n=== VALIDACIÓN MANUAL DE DATOS (muestra de 10) ===")

    sample = df.sample(10, random_state=42)[
        ["Ticket Subject", "Ticket Description", "Target"]
    ]

    for idx, row in sample.iterrows():
        print(f"\n--- Ejemplo {idx} ---")
        print(f"Asunto: {row['Ticket Subject']}")
        print(f"Descripción: {row['Ticket Description'][:100]}...")
        print(f"Etiqueta: {row['Target']}")
        print("-" * 50)


if __name__ == "__main__":
    # Cargar datos
    df_cleaned = pd.read_csv("data/processed/tickets_cleaned-2.csv")

    # 1. Investigación profunda
    df_cleaned = deep_data_investigation(df_cleaned)

    # 2. Validación manual
    manual_data_validation(df_cleaned)

    # 3. Verificar consistencia de etiquetas
    check_label_consistency(df_cleaned)

    # 4. Crear y probar categorías simplificadas
    df_simplified = create_simplified_categories(df_cleaned)
    simplified_model = train_simplified_model(df_simplified)

    # 5. Si las categorías simplificadas funcionan, guardar el nuevo dataset
    df_simplified.to_csv("data/processed/tickets_simplified.csv", index=False)
    print("\n✅ Dataset con categorías simplificadas guardado")
