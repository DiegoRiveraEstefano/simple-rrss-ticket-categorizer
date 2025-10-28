import joblib
import pandas as pd
from api.v1.schemas import Ticket

pipeline = joblib.load("models/simple_ticket_classifier.pkl")


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
    prediction = pipeline.predict(df)
    return prediction[0]