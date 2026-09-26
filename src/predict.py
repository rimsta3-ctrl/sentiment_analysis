"""
predict.py

CLI for predicting sentiment on new text using a trained model.

Usage:
    python src/predict.py --model baseline --text "This movie was great!"
    python src/predict.py --model transformer --model-dir models/distilbert_sentiment --text "..."
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text

LABELS = {0: "negative", 1: "positive"}


def predict_baseline(text: str, model_path: str = "models/baseline_tfidf_logreg.joblib"):
    import joblib
    pipeline = joblib.load(model_path)
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    return LABELS[pred], float(proba[pred])


def predict_transformer(text: str, model_dir: str = "models/distilbert_sentiment"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(torch.argmax(probs))
    return LABELS[pred], float(probs[pred])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--model-path", default="models/baseline_tfidf_logreg.joblib")
    parser.add_argument("--model-dir", default="models/distilbert_sentiment")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    if args.model == "baseline":
        label, confidence = predict_baseline(args.text, args.model_path)
    else:
        label, confidence = predict_transformer(args.text, args.model_dir)

    print(f"Text: {args.text}")
    print(f"Prediction: {label} (confidence: {confidence:.3f})")
