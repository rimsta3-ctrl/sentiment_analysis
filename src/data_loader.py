"""
data_loader.py

Handles loading data for the sentiment analysis project.

Supports three sources:
1. "sample"  -> the small bundled CSV (data/sample_reviews.csv), good for
                quickly testing that the pipeline runs end-to-end.
2. a CSV path -> any CSV with 'text' and 'label' columns (label in
                {"positive", "negative"} or {0,1}).
3. "imdb"    -> the full 50k-review IMDB dataset via HuggingFace `datasets`.
                Requires internet access (downloads from huggingface.co).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"negative": 0, "positive": 1, 0: 0, 1: 1}


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].map(lambda x: LABEL_MAP.get(x, x))
    df["label"] = df["label"].astype(int)
    return df


def load_csv(path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"
    df = _normalize_labels(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_sample(test_size: float = 0.25, random_state: int = 42):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "sample_reviews.csv")
    return load_csv(path, test_size=test_size, random_state=random_state)


def load_imdb(sample_size: int = None):
    """
    Loads the full IMDB dataset via HuggingFace `datasets`.
    Requires internet access to huggingface.co (not available in
    network-restricted sandboxes -- run this locally or in Colab).
    """
    from datasets import load_dataset

    ds = load_dataset("imdb")
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    if sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(sample_size // 4, len(test_df)), random_state=42).reset_index(drop=True)

    return train_df, test_df


def load_twitter(test_size: float = 0.2, random_state: int = 42):
    """
    Loads NLTK's `twitter_samples` corpus: 5,000 real positive tweets and
    5,000 real negative tweets. Downloads automatically on first call
    (requires internet access to raw.githubusercontent.com).
    """
    import nltk
    nltk.download("twitter_samples", quiet=True)
    from nltk.corpus import twitter_samples

    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")

    df = pd.DataFrame(
        {"text": pos + neg, "label": [1] * len(pos) + [0] * len(neg)}
    )
    df = _normalize_labels(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_data(source: str = "sample", **kwargs):
    """
    source: "sample" | "imdb" | "twitter" | "/path/to/file.csv"
    """
    if source == "sample":
        return load_sample(**kwargs)
    elif source == "imdb":
        return load_imdb(**kwargs)
    elif source == "twitter":
        return load_twitter(**kwargs)
    elif os.path.isfile(source):
        return load_csv(source, **kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")


if __name__ == "__main__":
    train_df, test_df = load_data("sample")
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(train_df.head())
