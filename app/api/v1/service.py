import joblib
import pandas as pd
from api.v1.schemas import Ticket

pipeline = joblib.load("ticket_classifier_pipeline.pkl")


def predict_ticket_category(ticket: Ticket) -> str:
    """
    Predicts the category of a ticket.

    Args:
        ticket: The ticket data.

    Returns:
        The predicted category.
    """
    data = {
        "Ticket Subject": [ticket.ticket_subject],
        "Ticket Description": [ticket.ticket_description],
        "Customer Age": [ticket.customer_age],
        "Customer Gender": [ticket.customer_gender],
        "Product Purchased": [ticket.product_purchased],
        "Ticket Channel": [ticket.ticket_channel],
        "Ticket Priority": [ticket.ticket_priority],
        "Ticket Status": [ticket.ticket_status],
        "Customer Satisfaction Rating": [ticket.customer_satisfaction_rating],
    }
    df = pd.DataFrame(data)
    prediction = pipeline.predict(df)
    return prediction[0]
