"""
transformer_model.py

Fine-tunes a pretrained DistilBERT model for binary sentiment classification.

Requires internet access to download the pretrained weights from
huggingface.co, and ideally a GPU for reasonable training times.
Run this locally or in Google Colab (free GPU) -- not inside a
network-restricted sandbox.

Usage:
    python src/transformer_model.py --source imdb --epochs 2
    python src/transformer_model.py --source sample --epochs 3   # quick smoke test
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

MODEL_NAME = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_transformer(source: str = "imdb", epochs: int = 2, batch_size: int = 16,
                       model_out_dir: str = "models/distilbert_sentiment",
                       max_length: int = 256):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )

    train_df, test_df = load_data(source)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/transformer_ckpts",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    os.makedirs(model_out_dir, exist_ok=True)
    trainer.save_model(model_out_dir)
    tokenizer.save_pretrained(model_out_dir)
    print(f"Model saved to {model_out_dir}")

    return trainer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="imdb", help="'imdb', 'sample', or a CSV path")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-out-dir", default="models/distilbert_sentiment")
    args = parser.parse_args()

    train_transformer(source=args.source, epochs=args.epochs,
                       batch_size=args.batch_size, model_out_dir=args.model_out_dir)
