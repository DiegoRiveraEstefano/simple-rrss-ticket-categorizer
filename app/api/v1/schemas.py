from pydantic import BaseModel


class Ticket(BaseModel):
    """
    Schema for ticket prediction input.
    """

    ticket_subject: str
    ticket_description: str


class Prediction(BaseModel):
    """
    Schema for the prediction output.
    """

    category: str
