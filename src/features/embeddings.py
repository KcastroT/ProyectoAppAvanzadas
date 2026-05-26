"""BETO (Spanish BERT) sentence embeddings as a feature extractor.

This module wraps a pretrained BETO model so it can be used as a drop-in
alternative to TF-IDF: each text is encoded into a single dense vector by
mean-pooling the last hidden states (masking out padding tokens).

The model is used in inference mode only (no fine-tuning), which makes it a
lightweight upgrade path from classic bag-of-words features toward
transformer-based representations, while still feeding a classic classifier
such as LinearSVC.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BetoEmbedder:
    """Generate dense sentence embeddings using a frozen BETO model."""

    def __init__(
        self,
        model_name,
        max_length=128,
        batch_size=32,
        device=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name)

        self.max_length = max_length

        self.batch_size = batch_size

        self.device = device or self._auto_device()

        self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def _auto_device():
        """Pick the best available device (CUDA > Apple MPS > CPU)."""
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """Average token embeddings, ignoring padding positions."""
        mask = attention_mask.unsqueeze(-1).float()

        summed = (last_hidden_state * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    @torch.no_grad()
    def transform(self, texts):
        """Encode an iterable of texts into a (n_samples, hidden_size) array."""
        texts = [str(text) for text in texts]

        embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)

            pooled = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings)


def build_embedder(model_name, max_length=128, batch_size=32):
    """Create a BETO embedder."""
    return BetoEmbedder(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )
