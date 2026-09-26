"""
baseline_model.py

TF-IDF + Logistic Regression baseline for sentiment classification.
Fast to train, interpretable, and a solid sanity-check before the
transformer model.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_series
from data_loader import load_data


def build_pipeline(max_features: int = 20000, ngram_range=(1, 2), C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                   sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, C=C, class_weight="balanced")),
    ])


def plot_learning_curve(pipeline, X, y, out_path: str = None, cv: int = 5):
    """
    Trains the pipeline on increasing subsets of the training data and plots
    train vs. cross-validation score, to check for over/underfitting and
    whether more data would likely help.
    """
    # Keep cv small enough, and the smallest training slice large enough,
    # that every fold still sees both classes (matters for tiny datasets).
    min_class_count = min(np.bincount(y))
    cv = max(2, min(cv, min_class_count))
    min_train_frac = min(1.0, max(0.3, 20 / len(X)))

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring="f1",
        train_sizes=np.linspace(min_train_frac, 1.0, 6),
        n_jobs=-1, random_state=42,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, "o-", label="Training F1")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(train_sizes, val_mean, "o-", label="Cross-val F1")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Learning curve saved to {out_path}")
    plt.close(fig)

    return train_sizes, train_mean, val_mean


def train_baseline(source: str = "sample", model_out: str = None,
                    val_size: float = 0.15, plot_curve: bool = True,
                    curve_out: str = "outputs/learning_curve.png"):
    train_df, test_df = load_data(source)

    # Extra split off a validation set from the training data, so the test
    # set stays untouched until final evaluation.
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=42, stratify=train_df["label"]
    )

    is_twitter = source == "twitter"
    clean_kwargs = {"remove_mentions": True, "remove_hashtag_symbol": True} if is_twitter else {}

    X_train = clean_series(train_df["text"], **clean_kwargs)
    y_train = train_df["label"]
    X_val = clean_series(val_df["text"], **clean_kwargs)
    y_val = val_df["label"]
    X_test = clean_series(test_df["text"], **clean_kwargs)
    y_test = test_df["label"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_preds = pipeline.predict(X_val)
    print("--- Validation set ---")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f} | F1: {f1_score(y_val, val_preds):.4f}")

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("--- Test set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    if plot_curve:
        # Learning curve is computed on train+val combined via internal CV folds.
        import pandas as pd
        X_curve = clean_series(
            pd.concat([train_df["text"], val_df["text"]], ignore_index=True), **clean_kwargs
        )
        y_curve = pd.concat([y_train, y_val], ignore_index=True)
        plot_learning_curve(build_pipeline(), X_curve, y_curve, out_path=curve_out)

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(pipeline, model_out)
        print(f"Model saved to {model_out}")

    return pipeline, {"val_accuracy": accuracy_score(y_val, val_preds), "accuracy": acc, "f1": f1}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sample", help="'sample', 'imdb', 'twitter', or a CSV path")
    parser.add_argument("--model-out", default="models/baseline_tfidf_logreg.joblib")
    parser.add_argument("--no-curve", action="store_true", help="skip the learning curve plot")
    args = parser.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_out = os.path.join(here, args.model_out) if not os.path.isabs(args.model_out) else args.model_out
    curve_out = os.path.join(here, "outputs/learning_curve.png")

    train_baseline(source=args.source, model_out=model_out,
                   plot_curve=not args.no_curve, curve_out=curve_out)
