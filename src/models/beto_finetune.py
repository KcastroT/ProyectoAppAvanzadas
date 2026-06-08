"""End-to-end fine-tuning of BETO for Spanish text classification.

Unlike ``features.embeddings.BetoEmbedder``, which uses BETO as a frozen
feature extractor, this module attaches a linear classification head on top
of BETO and updates *all* weights via gradient descent on the labeled data.

The :class:`BetoClassifier` exposes a small scikit-learn-style API
(``fit`` / ``predict`` / ``predict_proba`` / ``classes_``) so it slots into
the same evaluation harness used by the SVM models.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


class _TextDataset(Dataset):
    """Tokenize one example at a time so the collator can pad per batch."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )

        item = {key: value for key, value in encoding.items()}

        if self.labels is not None:
            item["labels"] = int(self.labels[idx])

        return item


class BetoClassifier(BaseEstimator, ClassifierMixin):
    """Fine-tune BETO with a classification head, sklearn-style."""

    def __init__(
        self,
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        max_length=128,
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=42,
        output_dir="./beto_ft_runs",
        verbose=True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.output_dir = output_dir
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        set_seed(self.seed)

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)

        y_array = np.asarray(y)

        self.classes_ = np.array(sorted(np.unique(y_array)))

        self._label_to_idx = {
            label: idx for idx, label in enumerate(self.classes_)
        }

        self._idx_to_label = {
            idx: label for label, idx in self._label_to_idx.items()
        }

        y_idx = np.array(
            [self._label_to_idx[value] for value in y_array],
            dtype=np.int64,
        )

        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes_),
        )

        train_dataset = _TextDataset(
            texts=list(X),
            labels=y_idx,
            tokenizer=self.tokenizer_,
            max_length=self.max_length,
        )

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            seed=self.seed,
            disable_tqdm=not self.verbose,
        )

        trainer = Trainer(
            model=self.model_,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer_),
            processing_class=self.tokenizer_,
        )

        trainer.train()

        self.model_.eval()

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _logits(self, X):
        texts = [str(t) for t in X]

        device = next(self.model_.parameters()).device

        all_logits = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            encoded = self.tokenizer_(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            outputs = self.model_(**encoded)

            all_logits.append(outputs.logits.cpu().numpy())

        return np.vstack(all_logits)

    def predict(self, X):
        logits = self._logits(X)

        predicted_indices = logits.argmax(axis=1)

        return np.array(
            [self._idx_to_label[idx] for idx in predicted_indices]
        )

    def predict_proba(self, X):
        logits = self._logits(X)

        shifted = logits - logits.max(axis=1, keepdims=True)

        exp_logits = np.exp(shifted)

        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def build_model(
    model_name,
    max_length=128,
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=42,
    output_dir="./beto_ft_runs",
):
    """Create a fine-tuning BETO classifier."""
    return BetoClassifier(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=seed,
        output_dir=output_dir,
    )
