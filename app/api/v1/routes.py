from api.v1 import schemas
from api.v1 import service
from fastapi import APIRouter

router = APIRouter()


@router.post("/predict", response_model=schemas.Prediction)
def predict(ticket: schemas.Ticket):
    """
    Predicts the category of a ticket.
    """
    category = service.predict_ticket_category(ticket)
    return schemas.Prediction(category=category)
