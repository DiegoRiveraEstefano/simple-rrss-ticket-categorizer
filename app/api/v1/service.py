import joblib
import numpy as np
import pandas as pd

from app.api.v1.schemas import Ticket


def tokenize_text(text, amount=2):
    """Preprocesa el texto para que sea compatible con el modelo."""
    return " ".join(
        [t for t in text.split() if len(t) > amount],
    )


pipeline = joblib.load("models/simple_ticket_classifier.pkl")


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


def predict_ticket_category(ticket: Ticket) -> str:
    """
    Predicts the category of a ticket.

    Args:
        ticket: The ticket data.

    Returns:
        The predicted category.
    """
    data = {
        "subject": [ticket.ticket_subject],
        "body": [ticket.ticket_description],
    }
    df = pd.DataFrame(data)
    df = add_text_features(df)
    prediction = pipeline.predict(df)
    return prediction[0]
