"""Zero-/few-shot tweet classification with a local LLM via Ollama.

This wraps a locally served instruction-tuned LLM (e.g. ``llama3.2:3b``) in a
scikit-learn compatible classifier so it can be dropped into the same
evaluation harness as the TF-IDF / BETO models.

Unlike the other estimators, this model is **not trained**: ``fit`` only records
the label set and (optionally) picks a handful of labeled examples from the
training data to include in the prompt (few-shot). ``predict`` then asks the LLM
to label each text, parsing a JSON ``{"label": ..., "prob_anorexia": ...}``
response; the self-reported probability also powers ``predict_proba`` (and thus
ROC/AUC), useful for ranking but only loosely calibrated.

It talks to Ollama's local REST API (http://localhost:11434) using only the
Python standard library, so it adds no extra dependencies. Make sure Ollama is
running and the model is pulled::

    ollama pull llama3.2:3b
    ollama serve   # usually already running as a background service
"""

import json
import urllib.error
import urllib.request

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class OllamaClassifier(BaseEstimator, ClassifierMixin):
    """Classify text by prompting a local LLM served by Ollama.

    Args:
        model_name: Ollama model tag (e.g. ``"llama3.2:3b"``).
        labels: Iterable of valid class labels the LLM must choose from.
        task_description: One-line description of the classification task,
            written in Spanish since the corpus is Spanish.
        n_few_shot: Number of labeled examples per class to embed in the prompt.
            ``0`` means zero-shot.
        host: Base URL of the Ollama server.
        temperature: Decoding temperature (0 = deterministic, best for labels).
        max_chars: Truncate each tweet to this many characters in the prompt.
        timeout: Per-request timeout in seconds.
        verbose: Print progress every ``verbose`` predictions (0 = silent).
    """

    def __init__(
        self,
        model_name="llama3.2:3b",
        labels=("anorexia", "control"),
        task_description=(
            "Clasifica el siguiente tuit en español segun si su contenido "
            "promueve, expresa o gira en torno a un trastorno de la conducta "
            "alimentaria (anorexia) o no (control)."
        ),
        n_few_shot=4,
        host="http://localhost:11434",
        temperature=0.0,
        max_chars=500,
        timeout=120,
        verbose=50,
    ):
        self.model_name = model_name
        self.labels = labels
        self.task_description = task_description
        self.n_few_shot = n_few_shot
        self.host = host
        self.temperature = temperature
        self.max_chars = max_chars
        self.timeout = timeout
        self.verbose = verbose

    # =========================
    # sklearn API
    # =========================

    def fit(self, X, y):
        """Record the label set and pick few-shot examples (no training)."""
        y = np.asarray(y)

        # classes_ is required by the evaluation harness (metrics.py).
        self.classes_ = np.array(sorted(set(self.labels)))

        # Positive class for the self-reported probability / AUC. classes_ is
        # sorted, so classes_[0] == "anorexia" (the condition to detect).
        self._fallback_label = self.classes_[0]
        self._positive_label = self.classes_[0]

        # predict() and predict_proba() share one query pass via this cache.
        self._cache_key = None
        self._cache = None

        self._few_shot = self._select_few_shot(np.asarray(X, dtype=object), y)

        return self

    def predict(self, X):
        """Label each text by querying the LLM."""
        return self._query_all(X)[0]

    def predict_proba(self, X):
        """Return ``[P(anorexia), P(control)]`` per text (columns match classes_).

        The probabilities are the LLM's own self-reported confidence, which
        enables ROC/AUC for the LLM rows. They are useful for *ranking* (AUC)
        but only loosely calibrated.
        """
        return self._query_all(X)[1]

    def _query_all(self, X):
        """Query each text once; return ``(labels, proba)`` and cache them.

        ``predict`` and ``predict_proba`` both call this; caching on the exact
        input avoids re-querying the LLM when the harness asks for both.
        """
        texts = [str(text) for text in X]

        key = tuple(texts)

        if self._cache_key == key:
            return self._cache

        labels = []
        probs = []

        for idx, text in enumerate(texts, start=1):
            label, p_pos = self._classify_one(text)

            labels.append(label)
            probs.append(p_pos)

            if self.verbose and idx % self.verbose == 0:
                print(f"    [LLM] {idx}/{len(texts)} classified")

        p_pos = np.asarray(probs, dtype=float)

        # Columns ordered to match classes_ == [positive, other].
        proba = np.column_stack([p_pos, 1.0 - p_pos])

        self._cache_key = key
        self._cache = (np.array(labels), proba)

        return self._cache

    # =========================
    # Prompt construction
    # =========================

    def _select_few_shot(self, X, y):
        """Pick up to ``n_few_shot`` examples per class for the prompt."""
        if self.n_few_shot <= 0:
            return []

        examples = []

        rng = np.random.default_rng(42)

        for label in self.classes_:
            idx = np.where(y == label)[0]

            if len(idx) == 0:
                continue

            chosen = rng.choice(
                idx,
                size=min(self.n_few_shot, len(idx)),
                replace=False,
            )

            for i in chosen:
                examples.append((self._truncate(str(X[i])), str(label)))

        rng.shuffle(examples)

        return examples

    def _truncate(self, text):
        text = " ".join(text.split())

        if len(text) > self.max_chars:
            return text[: self.max_chars]

        return text

    def _build_messages(self, text):
        label_list = ", ".join(f'"{c}"' for c in self.classes_)

        system = (
            f"{self.task_description}\n"
            f"Responde unicamente con un objeto JSON con la forma "
            f'{{"label": <etiqueta>, "prob_anorexia": <numero>}}, donde '
            f"<etiqueta> es exactamente uno de: {label_list}, y <numero> es la "
            f'probabilidad (entre 0 y 1) de que el tuit sea de la clase '
            f'"{self._positive_label}". No agregues explicaciones.'
        )

        messages = [{"role": "system", "content": system}]

        for ex_text, ex_label in self._few_shot:
            messages.append({"role": "user", "content": f"Tuit: {ex_text}"})

            ex_prob = (
                0.95 if str(ex_label) == str(self._positive_label) else 0.05
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"label": ex_label, "prob_anorexia": ex_prob}
                    ),
                }
            )

        messages.append(
            {"role": "user", "content": f"Tuit: {self._truncate(text)}"}
        )

        return messages

    # =========================
    # Ollama transport
    # =========================

    def _classify_one(self, text):
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(text),
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            content = self._post_chat(payload)

            label = self._parse_label(content)
            prob = self._parse_prob(content, label)

            return label, prob
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            # Never let a single bad call abort the whole comparison.
            fallback = self._fallback_label

            return fallback, self._prob_from_label(fallback)

    def _post_chat(self, payload):
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["message"]["content"]

    def _parse_label(self, content):
        """Map the LLM's JSON response to one of the known labels."""
        parsed = json.loads(content)

        raw = str(parsed.get("label", "")).strip().lower()

        for label in self.classes_:
            if str(label).lower() == raw:
                return label

        # Lenient fallback: substring match (e.g. "anorexia (TCA)").
        for label in self.classes_:
            if str(label).lower() in raw:
                return label

        return self._fallback_label

    def _parse_prob(self, content, label):
        """Extract the self-reported P(positive class) from the JSON response.

        Falls back to a label-consistent value when the field is missing or out
        of range, so the probability (and thus AUC) stays well-defined.
        """
        try:
            parsed = json.loads(content)

            value = float(parsed.get("prob_anorexia"))

            if 0.0 <= value <= 1.0:
                return value
        except (TypeError, ValueError, AttributeError):
            pass

        return self._prob_from_label(label)

    def _prob_from_label(self, label):
        """Coarse P(positive) implied by a hard label (no probability given)."""
        return 0.85 if str(label) == str(self._positive_label) else 0.15


def build_model(
    model_name="llama3.2:3b",
    labels=("anorexia", "control"),
    n_few_shot=4,
):
    """Create an Ollama-backed LLM classifier."""
    return OllamaClassifier(
        model_name=model_name,
        labels=labels,
        n_few_shot=n_few_shot,
    )
