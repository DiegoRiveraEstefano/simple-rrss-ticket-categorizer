import re

import joblib
import pandas as pd


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
    # cargar modelo
    model = joblib.load("models/simple_ticket_classifier.pkl")

    # 5. Probar con ejemplos
    test_cases = [
        "Problema crítico del servidor requiere atención inmediata",
        "Consulta sobre disponibilidad de producto",
        "Error en la facturación del servicio",
        "Solicitud de cambio de configuración",
        "Problema técnico con el software",
        "@[EmpresaX] I received the bill and the amount is incorrect. They are charging me for a plan that I canceled last month.",
        "@[EmpresaX] hola, quiero portarme a su compañía, ¿dónde veo los planes de fibra óptica??",
        "@[EmpresaX] otra vez caídos? qué servicio más malo... como siempre.",
    ]

    print("\n=== PRUEBAS RÁPIDAS ===")
    for test_case in test_cases:
        predict_ticket_simple(model, test_case)
