from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Ticket Sentiment Service",
    description="PCAM ZC412 Mini-Project-1 Flavor C — support ticket / review sentiment classifier",
    version="0.1.0",
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    channel: str | None = None


class TextResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ticket-sentiment", "model_loaded": False}


@app.post("/predict", response_model=TextResponse)
def predict(payload: TextRequest) -> TextResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    raise NotImplementedError("Week 3: load best_model.joblib and classify text")
