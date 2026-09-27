"""
api.py

A small FastAPI demo exposing the trained model(s) for live predictions.

Run:
    uvicorn src.api:app --reload --port 8000

Then either open http://localhost:8000/docs for the interactive Swagger UI,
or:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"text": "This movie was absolutely incredible!"}'
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(HERE, "models", "baseline_tfidf_logreg.joblib")
TRANSFORMER_DIR = os.path.join(HERE, "models", "distilbert_sentiment")

LABELS = {0: "negative", 1: "positive"}

models = {}  # populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load whichever models are available on disk ---
    if os.path.exists(BASELINE_PATH):
        import joblib
        models["baseline"] = joblib.load(BASELINE_PATH)
        print(f"Loaded baseline model from {BASELINE_PATH}")
    else:
        print(f"No baseline model found at {BASELINE_PATH} (train one with src/baseline_model.py)")

    if os.path.isdir(TRANSFORMER_DIR):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        models["transformer_tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        models["transformer_model"] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        models["transformer_model"].eval()
        print(f"Loaded transformer model from {TRANSFORMER_DIR}")
    else:
        print(f"No transformer model found at {TRANSFORMER_DIR} (train one with src/transformer_model.py)")

    yield
    models.clear()


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str
    model: str = "transformer"  # "baseline" or "transformer"


class PredictResponse(BaseModel):
    text: str
    model_used: str
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model == "baseline":
        if "baseline" not in models:
            raise HTTPException(status_code=503, detail="Baseline model not loaded/trained yet.")
        cleaned = clean_text(req.text)
        pipeline = models["baseline"]
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        return PredictResponse(text=req.text, model_used="baseline",
                                label=LABELS[pred], confidence=float(proba[pred]))

    elif req.model == "transformer":
        if "transformer_model" not in models:
            raise HTTPException(status_code=503, detail="Transformer model not loaded/trained yet.")
        import torch
        tokenizer = models["transformer_tokenizer"]
        model = models["transformer_model"]
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(probs))
        return PredictResponse(text=req.text, model_used="transformer",
                                label=LABELS[pred], confidence=float(probs[pred]))

    else:
        raise HTTPException(status_code=400, detail="model must be 'baseline' or 'transformer'")
