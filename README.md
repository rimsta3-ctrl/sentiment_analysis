# NLP Sentiment Analysis

Binary sentiment classification (positive / negative) on movie reviews, with
two models so you can compare a fast classic baseline against a fine-tuned
transformer:

1. **Baseline** — TF-IDF + Logistic Regression (`src/baseline_model.py`)
2. **Main model** — Fine-tuned DistilBERT (`src/transformer_model.py`)

## Project structure

```
sentiment_project/
├── data/
│   └── sample_reviews.csv       # tiny bundled dataset for smoke-testing
├── models/                      # trained models get saved here
├── outputs/                     # checkpoints, plots, etc.
├── src/
│   ├── data_loader.py           # loads sample / IMDB / Twitter / custom CSV
│   ├── preprocessing.py         # text cleaning
│   ├── baseline_model.py        # TF-IDF + LogisticRegression + learning curve
│   ├── transformer_model.py     # DistilBERT fine-tuning (HuggingFace)
│   ├── evaluate.py              # metrics + confusion matrix plotting
│   ├── predict.py               # CLI inference on new text
│   └── api.py                   # FastAPI demo endpoint
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Baseline model (TF-IDF + Logistic Regression)

Runs instantly on CPU, no internet required beyond installing packages.

```bash
# Quick smoke test on the bundled 40-row sample
python src/baseline_model.py --source sample

# Full run on IMDB (auto-downloads via HuggingFace `datasets`)
python src/baseline_model.py --source imdb --model-out models/baseline_tfidf_logreg.joblib

# Full run on real tweets (10k labeled tweets via NLTK's twitter_samples)
python src/baseline_model.py --source twitter --model-out models/baseline_twitter.joblib

# Or point it at your own CSV (columns: text, label)
python src/baseline_model.py --source path/to/your_data.csv
```

Expect roughly **88-90% accuracy** on full IMDB, and **~77%** on the Twitter
data (tweets are short/noisy, so this is a normal, honest result for a
linear baseline — the transformer model closes much of that gap).

Each run automatically:
- carves out a validation set (separate from the test set) and reports
  validation metrics before the final test evaluation,
- saves a learning curve plot to `outputs/learning_curve.png` (train vs.
  cross-val F1 as training size grows) — useful for spotting over/underfitting
  and judging whether more data would help.

## 2. Transformer model (DistilBERT)

Needs internet access (downloads pretrained weights from huggingface.co) and
ideally a GPU. **Run this locally or in Google Colab** — not inside a
network-restricted environment.

```bash
# Quick smoke test on the sample data (few seconds, low accuracy expected)
python src/transformer_model.py --source sample --epochs 3

# Full fine-tuning run on IMDB (~20-40 min on a Colab T4 GPU)
python src/transformer_model.py --source imdb --epochs 2 --batch-size 16
```

Expect roughly **92-94% accuracy** on the full IMDB test set — a solid
improvement over the baseline.

### Colab tip
If you don't have a local GPU, open a new Google Colab notebook, set
Runtime → Change runtime type → GPU, `!pip install -r requirements.txt`,
upload this folder (or `git clone` it), and run the command above.

## 3. Predicting on new text

```bash
# Using the baseline model
python src/predict.py --model baseline --text "This movie was absolutely incredible!"

# Using the fine-tuned transformer
python src/predict.py --model transformer --text "This movie was absolutely incredible!"
```

## 4. Live demo API (FastAPI)

Serves whichever model(s) it finds already trained in `models/`.

```bash
uvicorn src.api:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for an interactive Swagger UI, or:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely incredible!", "model": "baseline"}'
```

`model` can be `"baseline"` or `"transformer"` (the latter only works once
you've run `transformer_model.py` and it's saved to `models/distilbert_sentiment`).
`GET /health` reports which models are currently loaded.

## 5. Evaluation / confusion matrix

`src/evaluate.py` exposes `full_report(y_true, y_pred, out_path=...)` which
prints precision/recall/F1 and saves a confusion matrix plot. Call it from
your own script after generating predictions, e.g.:

```python
from src.evaluate import full_report
full_report(y_test, preds, out_path="outputs/confusion_matrix.png")
```

## Notes on the dataset

- **Sample data** (`data/sample_reviews.csv`): 40 short hand-written reviews,
  useful only for confirming the code runs end-to-end — accuracy on it is
  not meaningful.
- **Full IMDB**: 50,000 real movie reviews (25k train / 25k test, perfectly
  balanced pos/neg), the standard benchmark for this task. Loaded via
  `datasets.load_dataset("imdb")` — needs an internet connection.
- **Twitter** (`source="twitter"`): 10,000 real tweets (5k positive / 5k
  negative) from NLTK's `twitter_samples` corpus — downloads automatically
  on first use. Shorter, noisier, more informal text than IMDB reviews, so
  it's a good test of how well the pipeline generalizes beyond movie
  reviews. Tweet-specific cleaning (stripping @mentions, unwrapping
  `#hashtags`) is applied automatically when `source="twitter"`.
- Want a different domain still (e.g. product reviews)? Swap the `source`
  argument for any CSV with `text` and `label` columns — everything else in
  the pipeline stays the same.

## Suggested next steps / extensions
- Try `roberta-base` or `bert-base-uncased` instead of DistilBERT for a small
  accuracy bump (slower to train).
- Add SHAP or LIME explanations for the baseline model's predictions.
- Add authentication / rate limiting to `api.py` before deploying it anywhere
  public.
- Try training the transformer on the Twitter data too, to see how much of
  the baseline's domain gap it closes.
