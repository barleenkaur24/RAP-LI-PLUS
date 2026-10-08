from fastapi import FastAPI
from pydantic import BaseModel

from risk_detection.regex_detector import detect_pii
from risk_detection.risk_classifier import classify_entities


app = FastAPI(
    title="RAP-LI-PLUS Backend",
    description="Risk-aware privacy-preserving LLM inference backend",
    version="0.1.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "RAP-LI-PLUS backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze")
def analyze_text(request: TextRequest):
    """
    Detect PII in the input text and assign a risk level.
    """

    entities = detect_pii(request.text)
    classified_entities = classify_entities(entities)

    # Count entities by risk level
    risk_summary = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    for entity in classified_entities:
        risk = entity["risk"]
        risk_summary[risk] += 1

    return {
        "text": request.text,
        "entities": classified_entities,
        "count": len(classified_entities),
        "risk_summary": risk_summary
    }