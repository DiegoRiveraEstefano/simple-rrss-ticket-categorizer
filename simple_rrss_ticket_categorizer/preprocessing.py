import re

import numpy as np
import pandas as pd


# --- 1. Carga de Datos (Sin cambios) ---
def load_data(filepath):
    """Carga los datos desde un archivo CSV o Excel."""
    if filepath.endswith(".csv"):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error al leer CSV: {e}")
            raise ValueError("No se pudo cargar el archivo CSV.")
    elif filepath.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            print(f"Error al leer Excel: {e}")
            raise ValueError("No se pudo cargar el archivo Excel.")
    else:
        raise ValueError("Formato de archivo no soportado. Usa .csv o .xlsx")

    if df is None or df.empty:
        raise ValueError("El archivo está vacío o no contiene datos válidos.")
    print(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# --- 2. Manejo de Nulos (Sin cambios) ---
def handle_missing_values(df: pd.DataFrame):
    """Rellena valores faltantes con defaults apropiados."""
    print("Manejando valores faltantes...")
    # Columnas categóricas críticas
    df["type"] = df["type"].fillna("Unknown")
    df["queue"] = df["queue"].fillna("Unknown")
    df["priority"] = df["priority"].fillna("medium")
    df["language"] = df["language"].fillna("en")
    df["business_type"] = df["business_type"].fillna("Unknown")

    # Columnas de texto
    text_cols = ["subject", "body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Tags
    tag_columns = [f"tag_{i}" for i in range(1, 10)]
    for tag_col in tag_columns:
        if tag_col in df.columns:
            df[tag_col] = df[tag_col].fillna("")

    return df


# --- 3. Ingeniería de Features (Tu versión avanzada) ---
def add_text_features(df: pd.DataFrame):
    """
    Agrega un conjunto robusto de features numéricas derivadas del texto
    para mejorar la clasificación.
    """
    print("Extrayendo features numéricas del texto...")

    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)
    df["full_text"] = df["subject"] + " " + df["body"]

    # --- Features de Longitud y Ratio ---
    df["subject_length"] = df["subject"].str.len()
    df["body_length"] = df["body"].str.len()
    df["subject_word_count"] = df["subject"].str.split().str.len()
    df["body_word_count"] = df["body"].str.split().str.len()

    df["subject_uppercase_ratio"] = df["subject"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1),
    )
    df["body_uppercase_ratio"] = df["body"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1),
    )
    df["avg_word_length_body"] = np.where(
        df["body_word_count"] > 0,
        df["body_length"] / df["body_word_count"],
        0,
    )

    # --- Features de Puntuación y Caracteres Especiales ---
    df["has_question_mark"] = (
        df["full_text"].str.contains(r"\?", regex=True).astype(int)
    )
    df["has_exclamation"] = df["full_text"].str.contains(r"!", regex=True).astype(int)
    df["mentions_currency"] = (
        df["full_text"].str.contains(r"\$|€|£|CLP|USD", regex=True).astype(int)
    )
    df["is_social_media_mention"] = (
        df["full_text"].str.contains(r"\B@\w+", regex=True).astype(int)
    )
    df["has_digits"] = df["full_text"].str.contains(r"\d", regex=True).astype(int)

    # --- Features de Palabras Clave (Keywords mejoradas) ---
    support_keywords = (
        r"\b(error|fallo|problema|caído|no funciona|lento|servidor|software|bug|down|broken|404|503|crash|"
        r"failure|outage|offline|disconnect|no conecta|sin acceso|no puedo entrar|exception|timeout)\b"
    )
    billing_keywords = (
        r"\b(bill|billing|invoice|factura|boleta|cobro|pago|pagos|payment|charge|charged|cargos|"
        r"monto|amount|fee|cost|costo|costos|price|precio|refund|reembolso|overcharge|debit|credit|"
        r"tarjeta|saldo|deuda|abono|receipt|statement|recibo|dinero|money|compra|purchase)\b"
    )
    urgent_keywords = (
        r"\b(urgente|crítico|inmediato|asap|ahora|inmediatamente|emergencia|priority|prioridad|"
        r"urgent|critical|immediate|now)\b"
    )
    service_keywords = (
        r"\b(consulta|duda|info|información|portabilidad|planes|plan|contratar|ayuda|pregunta|horario|dirección|"
        r"question|query|help|information|assist|cómo|cuando|donde|what|how|where|when)\b"
    )
    management_keywords = (
        r"\b(solicitud|cambio|configuración|actualizar|modificar|install|upgrade|baja|cancelar|cancellation|"
        r"request|change|configure|update|modify|installation|alta|activar|desactivar|activate|deactivate)\b"
    )
    negative_sentiment_keywords = (
        r"\b(malo|pésimo|horrible|frustrado|molesto|rabia|basura|worst|terrible|queja|complaint|"
        r"decepcionado|enojado|harto|angry|frustrated|awful|sucks|bad|molestia|problemas)\b|"
        r"nunca funciona|siempre falla"
    )

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

    df = df.drop(columns=["full_text"])
    return df


def create_category_target(df: pd.DataFrame):
    """
    Crea una columna 'category' unificada basada en señales jerárquicas
    (queue > tags > type > business_type), con cobertura multilingüe y heurísticas contextualizadas.
    """

    print("🧩 Creando columna 'category' jerárquica optimizada...")

    # ======================================================
    # 1. Palabras clave multilingües agrupadas
    # ======================================================
    BILLING_KEYWORDS = re.compile(
        r"\b(billing|payment|invoice|refund|factura|cobro|overcharge|deuda|"
        r"facturation|paiement|remboursement|abrechnung|rechnung|zahlung|"
        r"pagamento|cobrança|fatura|reembolso|dinheiro|money|purchase|compra)\b",
        re.I,
    )

    TECH_KEYWORDS = re.compile(
        r"\b(tech|support|incident|bug|error|crash|network|server|maintenance|"
        r"falla|ca[ií]do|servicio|outage|system|hardware|software|performance|"
        r"panne|störung|problema|defeito|falha|offline)\b",
        re.I,
    )

    MGMT_KEYWORDS = re.compile(
        r"\b(manage|config|change|upgrade|install|activate|deactivate|cancel|"
        r"gestión|solicitud|modificar|configurar|setup|update|"
        r"anfrage|konfigurieren|configuração|solicitar)\b",
        re.I,
    )

    SERVICE_KEYWORDS = re.compile(
        r"\b(customer|service|sales|inquiry|consulta|question|help|account|"
        r"assist|pregunta|pedido|retour|devolu|vendas|aide|info|informação|"
        r"support_client|consult|ventes|helpdesk)\b",
        re.I,
    )

    # ======================================================
    # 2. Columnas relevantes (solo las existentes)
    # ======================================================
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    base_cols = ["queue", "type", "business_type"]

    for col in base_cols + tag_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.lower()

    # ======================================================
    # 3. Función auxiliar de detección (por regex)
    # ======================================================
    def detect_category(text):
        if BILLING_KEYWORDS.search(text):
            return "Billing"
        if TECH_KEYWORDS.search(text):
            return "Technical_Support"
        if MGMT_KEYWORDS.search(text):
            return "Service_Management"
        if SERVICE_KEYWORDS.search(text):
            return "Customer_Service"
        return None

    # ======================================================
    # 4. Inicialización vacía
    # ======================================================
    df["category"] = None

    # ======================================================
    # 5. Jerarquía de asignación
    # ======================================================

    # 5.1 Desde `queue` (prioridad máxima)
    df.loc[
        df["queue"].apply(lambda x: bool(BILLING_KEYWORDS.search(x))),
        "category",
    ] = "Billing"
    df.loc[df["queue"].apply(lambda x: bool(TECH_KEYWORDS.search(x))), "category"] = (
        "Technical_Support"
    )
    df.loc[df["queue"].apply(lambda x: bool(MGMT_KEYWORDS.search(x))), "category"] = (
        "Service_Management"
    )
    df.loc[
        df["queue"].apply(lambda x: bool(SERVICE_KEYWORDS.search(x))),
        "category",
    ] = "Customer_Service"

    # 5.2 Desde tags (prioridad media)
    for tag_col in tag_cols:
        mask = df["category"].isna() & df[tag_col].notna()
        df.loc[mask, "category"] = df.loc[mask, tag_col].apply(
            lambda x: detect_category(str(x)),
        )

    # 5.3 Desde `type` (prioridad baja)
    type_map = {
        "incident": "Technical_Support",
        "problem": "Customer_Service",
        "change": "Service_Management",
        "request": "Customer_Service",
    }
    df.loc[df["category"].isna(), "category"] = df["type"].map(type_map)

    # 5.4 Desde `business_type` (contexto residual)
    # Ejemplo: empresas de "Tech" → soporte técnico, "Store" → servicio o ventas
    df.loc[
        df["category"].isna()
        & df["business_type"].str.contains("store", case=False, na=False),
        "category",
    ] = "Customer_Service"
    df.loc[
        df["category"].isna()
        & df["business_type"].str.contains(
            "tech|it|software|consulting",
            case=False,
            na=False,
        ),
        "category",
    ] = "Technical_Support"

    # 5.5 Fallback final
    df["category"] = df["category"].fillna("Customer_Service")

    # ======================================================
    # 6. Limpieza final y resumen
    # ======================================================
    df["category"] = df["category"].astype("category")

    print("\n📊 Distribución final de categorías:")
    print(df["category"].value_counts())

    return df


# --- 5. Limpieza de Texto (Mejorada) ---
def clean_text_columns(df: pd.DataFrame):
    """
    Limpia columnas de texto: normaliza a minúsculas y quita espacios extra.
    Se ejecuta DESPUÉS de add_text_features para no borrar '?' o '$'.
    """
    print("Normalizando columnas de texto (minúsculas, espacios)...")
    text_cols = ["subject", "body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
            # Ya no borramos la puntuación aquí, eso lo hará el preprocesador del modelo
    return df


# --- 6. Pipeline de Preprocesamiento (Actualizado) ---
def preprocess_data(df: pd.DataFrame):
    """Función principal que aplica todas las transformaciones consolidadas."""
    print("Iniciando preprocesamiento de datos...")

    # 1. Llenar todos los valores NaN primero
    df = handle_missing_values(df)

    # 2. Correr la extracción de features MIENTRAS el texto aún tiene puntuación
    df = add_text_features(df)

    # 3. Crear el target final del modelo (¡NUEVA FUNCIÓN!)
    df = create_category_target(df)

    # 4. Correr la limpieza de texto (minúsculas, espacios) AL FINAL
    df = clean_text_columns(df)

    # Nota: Se eliminaron 'create_target_column' y 'encode_categorical_variables'
    # por ser redundantes o perjudiciales para el pipeline del modelo.

    print("Preprocesamiento completado.")
    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: list):
    """Elimina columnas innecesarias del DataFrame."""
    print(f"Eliminando columnas innecesarias: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


# --- 7. Ejecución Principal ---
if __name__ == "__main__":
    # Actualiza con la ruta correcta
    input_filepath = "data/raw/dataset-tickets-multi-lang3-4k.csv"
    output_filepath = "data/processed/tickets_cleaned_con_category.csv"

    try:
        df = load_data(input_filepath)

        print(df["queue"].value_counts())
        print(df["type"].value_counts())
        print(df["business_type"].value_counts())
        print(df["tag_1"].value_counts())

        df_cleaned = preprocess_data(df)

        # Eliminar columnas innecesarias
        df_cleaned = drop_columns(
            df_cleaned,
            [
                "type",
                "queue",
                "business_type",
                "answer",
                "priority",
                "language",
                "tag_1",
                "tag_2",
                "tag_3",
                "tag_4",
                "tag_5",
                "tag_6",
                "tag_7",
                "tag_8",
                "tag_9",
            ],
        )

        # Guardar el archivo limpio y eficiente
        df_cleaned.to_csv(output_filepath, index=False)

        print(f"\n✅ Datos guardados en '{output_filepath}'")
        print(f"Shape del dataset limpio: {df_cleaned.shape}")

        # Mostrar columnas clave para confirmar
        print("\nInformación del DataFrame limpio (columnas clave):")
        columnas_clave = [
            "subject",
            "body",
            "category",  # <-- Tu nueva columna target
            "mentions_billing",
            "mentions_error",
            "mentions_service",
        ]
        columnas_existentes = [
            col for col in columnas_clave if col in df_cleaned.columns
        ]
        print(df_cleaned[columnas_existentes].info())

        # Mostrar Distribucion de Categorias
        print("\nDistribución de Categorias:")
        print(df_cleaned["category"].value_counts())

    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Error en el procesamiento: {e}")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")
