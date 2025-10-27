from logging import basicConfig
from logging import getLogger

import joblib

from simple_rrss_ticket_categorizer.pipeline import MODEL_PATH
from simple_rrss_ticket_categorizer.pipeline import train_pipeline

basicConfig(level="INFO")
logger = getLogger(__name__)


def main():
    train_pipeline()
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado exitosamente.")
    logger.info(f"Modelo: {model}")

    # uso de modelo para predecir un ejemplo ficticio
    example = {
        "Priority": "High",
        "Issue_Type": "Software",
        "Submitter_Department": "IT",
        "Created_At": "2023-10-01 10:00:00",
        "Resolved_At": "2023-10-01 14:00:00",
        "Resolution_Time_Hours": 2.0,
    }


if __name__ == "__main__":
    main()
