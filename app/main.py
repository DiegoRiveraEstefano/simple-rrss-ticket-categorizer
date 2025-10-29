from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def tokenize_text(text, amount=2):
    """Preprocesa el texto para que sea compatible con el modelo."""
    return " ".join(
        [t for t in text.split() if len(t) > amount],
    )


from app.api.v1 import routes  # noqa: E402

app = FastAPI(
    title="Ticket Categorizer API",
    description="A simple API to categorize support tickets.",
    version="0.1.0",
)

app.include_router(routes.router, prefix="/api/v1")

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
