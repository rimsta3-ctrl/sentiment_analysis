"""
preprocessing.py

Text cleaning utilities.

Note: For the transformer model (DistilBERT), minimal cleaning is best --
the tokenizer handles casing, punctuation, etc. Heavy cleaning is mainly
useful for the classic TF-IDF baseline.
"""

import re


def clean_text(text: str, lowercase: bool = True, remove_html: bool = True,
                remove_punct: bool = False, remove_numbers: bool = False,
                remove_mentions: bool = False, remove_hashtag_symbol: bool = False) -> str:
    if remove_html:
        text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    if remove_mentions:
        text = re.sub(r"@\w+", " ", text)
    if remove_hashtag_symbol:
        text = re.sub(r"#(\w+)", r"\1", text)  # keep the word, drop the '#'
    if lowercase:
        text = text.lower()
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_series(series, **kwargs):
    """Apply clean_text to a pandas Series of texts. Pass e.g. remove_mentions=True for tweets."""
    return series.astype(str).apply(lambda t: clean_text(t, **kwargs))


if __name__ == "__main__":
    sample = "This movie was <br/>AMAZING!!! Check it out at http://example.com 10/10"
    print(clean_text(sample))
