"""Exploratory data figures for the report + best-model confusion matrix.

Run from ``src/``::

    python plot_eda.py

Produces in ``../reports/figures/``:
  1. eda_class_distribution.png  — class balance (anorexia vs control).
  2. eda_tweet_length.png        — tweet length (words) per class.
  3. eda_discriminative_terms.png — most class-distinctive words (log-odds).
  4. confusion_matrix_best.png   — confusion matrix of the best classical model
     (the one with the highest CV F1 in classical_cv_summary.json), fit on the
     full training set and evaluated on the external test set.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BETO_BATCH_SIZE,
    BETO_MAX_LENGTH,
    BETO_MODEL_NAME,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    SVM_C_BETO,
    TEST_FILE,
    TEXT_COLUMN,
    TRAIN_FILE,
    TRANSFORMER_TEXT_COLUMN,
)
from data.loader import load_csv_dataset, load_excel_dataset
from data.preprocessing import (
    add_clean_text_column,
    clean_text_light,
    fix_dataframe_encoding,
)
from features.embeddings import build_embedder
from features.vectorizer import build_vectorizer
from models.logistic_regression_model import build_model as build_lr
from models.random_forest_model import build_model as build_rf
from models.svm_model import build_model as build_svm

OUTPUT_DIR = Path("../reports/figures")

ANOREXIA_COLOR = "#d62728"
CONTROL_COLOR = "#1f77b4"


def prepare(df):
    """Fix encoding and add the cleaned + lightly-cleaned text columns."""
    df = fix_dataframe_encoding(df)
    df = add_clean_text_column(df, source_column=TEXT_COLUMN, target_column=CLEAN_TEXT_COLUMN)
    df = add_clean_text_column(
        df, source_column=TEXT_COLUMN, target_column=TRANSFORMER_TEXT_COLUMN,
        cleaner=clean_text_light,
    )
    return df


# =========================
# 1) Class distribution
# =========================

def plot_class_distribution(train_df):
    """Bar chart of the anorexia/control class balance in the training set."""
    counts = train_df[LABEL_COLUMN].value_counts()
    labels = list(counts.index)
    values = counts.values
    colors = [ANOREXIA_COLOR if l == "anorexia" else CONTROL_COLOR for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    total = values.sum()
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v}\n({100 * v / total:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Número de tuits")
    ax.set_title(f"Distribución de clases — entrenamiento (n = {total})")
    ax.set_ylim(0, values.max() * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "eda_class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


# =========================
# 2) Tweet length per class
# =========================

def plot_tweet_length(train_df):
    """Overlaid histograms of tweet length (in words) per class."""
    lengths = train_df[TEXT_COLUMN].astype(str).str.split().str.len()
    y = train_df[LABEL_COLUMN].to_numpy()

    a = lengths[y == "anorexia"]
    c = lengths[y == "control"]

    hi = int(np.percentile(lengths, 99))
    bins = np.arange(0, hi + 2, 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(a, bins=bins, alpha=0.6, color=ANOREXIA_COLOR,
            label=f"anorexia (mediana {a.median():.0f})", density=True)
    ax.hist(c, bins=bins, alpha=0.6, color=CONTROL_COLOR,
            label=f"control (mediana {c.median():.0f})", density=True)

    ax.set_xlabel("Longitud del tuit (número de palabras)")
    ax.set_ylabel("Densidad")
    ax.set_title("Longitud de los tuits por clase — entrenamiento")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = OUTPUT_DIR / "eda_tweet_length.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


# =========================
# 3) Most discriminative terms (log-odds)
# =========================

def plot_discriminative_terms(train_df, top_n=15):
    """Horizontal bars of the most class-distinctive words by log-odds ratio."""
    texts = train_df[TRANSFORMER_TEXT_COLUMN].astype(str).to_numpy()
    y = train_df[LABEL_COLUMN].to_numpy()

    vec = CountVectorizer(min_df=5, lowercase=True)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    counts_a = np.asarray(X[y == "anorexia"].sum(axis=0)).ravel()
    counts_c = np.asarray(X[y == "control"].sum(axis=0)).ravel()

    alpha = 1.0
    V = len(vocab)
    p_a = (counts_a + alpha) / (counts_a.sum() + alpha * V)
    p_c = (counts_c + alpha) / (counts_c.sum() + alpha * V)

    # Log-odds ratio: positive => distinctive of anorexia, negative => control.
    log_odds = np.log(p_a / (1 - p_a)) - np.log(p_c / (1 - p_c))

    order = np.argsort(log_odds)
    control_idx = order[:top_n]          # most negative
    anorexia_idx = order[-top_n:]        # most positive

    idx = np.concatenate([control_idx, anorexia_idx])
    words = vocab[idx]
    scores = log_odds[idx]
    colors = [ANOREXIA_COLOR if s > 0 else CONTROL_COLOR for s in scores]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(range(len(words)), scores, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Log-odds ratio  (← control      anorexia →)")
    ax.set_title("Términos más distintivos por clase (log-odds) — entrenamiento")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "eda_discriminative_terms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


# =========================
# 4) Confusion matrices (every model + a combined 2x3 grid)
# =========================

CM_MODELS = [
    ("TF-IDF + LinearSVC", "TF-IDF", "LinearSVC"),
    ("TF-IDF + RandomForest", "TF-IDF", "RandomForest"),
    ("TF-IDF + LogisticReg", "TF-IDF", "LogisticReg"),
    ("BETO + LinearSVC", "BETO", "LinearSVC"),
    ("BETO + RandomForest", "BETO", "RandomForest"),
    ("BETO + LogisticReg", "BETO", "LogisticReg"),
]

LABELS = ["anorexia", "control"]


def slug(name):
    """Filesystem-safe slug for a model name."""
    return name.lower().replace(" + ", "_").replace(" ", "").replace("-", "")


def build_classifier(features, clf_name):
    """Build the grid pipeline for a (representation, classifier) pair."""
    if clf_name == "RandomForest":
        clf = build_rf()
    elif clf_name == "LinearSVC":
        clf = build_svm(C=SVM_C_BETO) if features == "BETO" else build_svm()
    else:
        clf = build_lr()

    steps = []
    if features == "TF-IDF":
        steps.append(("tfidf", build_vectorizer()))
    elif clf_name in ("LinearSVC", "LogisticReg"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _draw_cm(y_test, y_pred, ax, title):
    """Draw one confusion matrix (counts) onto the given axes."""
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=LABELS, labels=LABELS,
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")


def plot_confusion_matrices(train_df, test_df):
    """Fit each of the 6 models and save its test confusion matrix + a 2x3 grid."""
    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_test = test_df[LABEL_COLUMN].to_numpy()

    X_tr_text = train_df[CLEAN_TEXT_COLUMN].to_numpy(dtype=object)
    X_te_text = test_df[CLEAN_TEXT_COLUMN].to_numpy(dtype=object)

    # BETO embeddings once (shared by the 3 BETO cells).
    embedder = build_embedder(
        model_name=BETO_MODEL_NAME, max_length=BETO_MAX_LENGTH,
        batch_size=BETO_BATCH_SIZE,
    )
    X_tr_beto = embedder.transform(train_df[TRANSFORMER_TEXT_COLUMN])
    X_te_beto = embedder.transform(test_df[TRANSFORMER_TEXT_COLUMN])

    # 2x3 grid (rows: TF-IDF, BETO; cols: SVM, RF, LogReg).
    grid_fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, (name, features, clf_name) in enumerate(CM_MODELS):
        if features == "TF-IDF":
            X_train, X_test = X_tr_text, X_te_text
        else:
            X_train, X_test = X_tr_beto, X_te_beto

        pipe = build_classifier(features, clf_name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"  {name:<24} test F1 = {f1:.3f}")

        # individual figure
        fig, ax = plt.subplots(figsize=(6, 5.5))
        _draw_cm(
            y_test, y_pred, ax,
            f"Matriz de confusión — {name}\n(test externo, n = {len(y_test)}, "
            f"F1 = {f1:.3f})",
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{slug(name)}.png", dpi=150)
        plt.close(fig)

        # grid cell
        _draw_cm(y_test, y_pred, axes.flat[i], f"{name}  (F1 = {f1:.3f})")

    grid_fig.suptitle(
        f"Matrices de confusión — modelos clásicos (test externo, n = {len(y_test)})",
        fontsize=14,
    )
    grid_fig.tight_layout()
    grid_fig.savefig(OUTPUT_DIR / "confusion_matrices_all.png", dpi=150)
    plt.close(grid_fig)
    print(f"-> {OUTPUT_DIR / 'confusion_matrices_all.png'}  (+ 6 individuales)")


def main():
    """Generate the EDA figures and the per-model confusion matrices."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_df = prepare(load_excel_dataset(TRAIN_FILE))
    test_df = prepare(load_csv_dataset(TEST_FILE))
    print(f"  train: {len(train_df)}  test: {len(test_df)}")

    print("\n=== EDA figures ===")
    plot_class_distribution(train_df)
    plot_tweet_length(train_df)
    plot_discriminative_terms(train_df)

    print("\n=== Confusion matrices (all models) ===")
    plot_confusion_matrices(train_df, test_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
