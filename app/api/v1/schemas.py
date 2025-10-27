from pydantic import BaseModel
from pydantic import Field


class Ticket(BaseModel):
    """
    Schema for ticket prediction input.
    """

    ticket_subject: str
    ticket_description: str
    customer_age: int | None = Field(default=0)
    customer_gender: str | None = "Unknown"
    product_purchased: str | None = "Unknown"
    ticket_channel: str | None = "Unknown"
    ticket_priority: str | None = "Unknown"
    ticket_status: str | None = "Unknown"
    customer_satisfaction_rating: int | None = Field(default=0)


class Prediction(BaseModel):
    """
    Schema for the prediction output.
    """

    category: str
