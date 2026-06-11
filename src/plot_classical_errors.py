"""Generalization diagnostics for the six classical models.

Everything here is grounded in **stratified 5-fold cross-validation** over the
full training set, using **weighted F1** (not accuracy) to avoid being misled
by class imbalance. No single train/test split is used as a quality signal.

Three deliverables (run from ``src/``: ``python plot_classical_errors.py``):

  1. Learning curves (per model) — train vs validation F1 as the training set
     grows, with ±1 std bands. Diagnoses bias (underfitting) vs variance
     (overfitting) and whether more data would help.
  2. Overfitting figure — grouped train vs CV F1 bars with the gap annotated
     (green <0.05, orange 0.05-0.10, red >0.10).
  3. Main comparison — mean CV F1 per model with std error bars, sorted
     best→worst. Answers "which model generalizes best to unseen data?".

Plus, as complementary detail, validation curves over the complexity knob
(``C`` for the linear models, ``max_depth`` for Random Forest) and a gap bar.

TF-IDF is vectorized inside the CV pipeline (refit on each fold, no leakage);
BETO embeddings come from a frozen, label-independent encoder, so they are
computed once and reused.

Figures are written to ``../reports/figures/``.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BETO_BATCH_SIZE,
    BETO_MAX_LENGTH,
    BETO_MODEL_NAME,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_STATE,
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
from evaluation.plots import (
    save_cv_vs_test,
    save_learning_curve_cv,
    save_model_comparison,
    save_overfit_gap_bar,
    save_overfit_synthesis,
    save_validation_curve,
)
from features.embeddings import build_embedder
from features.vectorizer import build_vectorizer
from models.logistic_regression_model import build_model as build_lr
from models.random_forest_model import build_model as build_rf
from models.svm_model import build_model as build_svm

OUTPUT_DIR = Path("../reports/figures")

# 5-fold stratified CV, shared by every diagnostic.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Learning-curve x-axis: fractions of the training set.
TRAIN_FRACTIONS = np.linspace(0.1, 1.0, 9)

# Complexity knob swept for the (complementary) validation curves.
C_VALUES = np.logspace(-3, 2, 8)
DEPTH_VALUES = [1, 2, 3, 5, 8, 12, 16, 24, None]

MODELS = [
    ("TF-IDF + LinearSVC", "TF-IDF", "svm"),
    ("TF-IDF + LogisticReg", "TF-IDF", "lr"),
    ("TF-IDF + RandomForest", "TF-IDF", "rf"),
    ("BETO + LinearSVC", "BETO", "svm"),
    ("BETO + LogisticReg", "BETO", "lr"),
    ("BETO + RandomForest", "BETO", "rf"),
]


def base_estimator(kind):
    """Fresh base classifier for the given kind."""
    if kind == "svm":
        return build_svm()
    if kind == "rf":
        return build_rf()
    return build_lr()


def make_pipeline_for(features, kind):
    """Pipeline whose final step is named 'clf' (so set_params can reach it).

    TF-IDF vectorizes raw text inside the pipeline (refit per fold); dense BETO
    embeddings are standardized before the linear models (trees skip scaling).
    """
    steps = []

    if features == "TF-IDF":
        steps.append(("tfidf", build_vectorizer()))
    elif kind in ("svm", "lr"):
        steps.append(("scaler", StandardScaler()))

    steps.append(("clf", base_estimator(kind)))

    return Pipeline(steps)


def slug(name):
    """Filesystem-safe slug for a model name."""
    return name.lower().replace(" + ", "_").replace(" ", "").replace("-", "")


def sweep_config(kind):
    """Return (param_name, values, xlabel, log_x, tick_labels) for the sweep."""
    if kind == "rf":
        labels = [str(v) if v is not None else "∞" for v in DEPTH_VALUES]
        return "clf__max_depth", DEPTH_VALUES, "max_depth", False, labels

    return (
        "clf__C",
        list(C_VALUES),
        "C  (inverso de la regularización)",
        True,
        None,
    )


def sweep_overfitting(features, kind, X, y, param_name, values):
    """Train vs CV F1 (5-fold) as the complexity parameter is swept."""
    tr_mean, tr_std, cv_mean, cv_std = [], [], [], []

    for value in values:
        pipe = make_pipeline_for(features, kind)
        pipe.set_params(**{param_name: value})

        res = cross_validate(
            pipe, X, y, cv=CV, scoring="f1_weighted", return_train_score=True
        )

        tr_mean.append(res["train_score"].mean())
        tr_std.append(res["train_score"].std())
        cv_mean.append(res["test_score"].mean())
        cv_std.append(res["test_score"].std())

    return (
        np.array(tr_mean),
        np.array(tr_std),
        np.array(cv_mean),
        np.array(cv_std),
    )


def cv_scores(features, kind, X, y):
    """Default-config 5-fold CV: (train_mean, cv_mean, cv_std)."""
    pipe = make_pipeline_for(features, kind)

    res = cross_validate(
        pipe, X, y, cv=CV, scoring="f1_weighted", return_train_score=True
    )

    return (
        res["train_score"].mean(),
        res["test_score"].mean(),
        res["test_score"].std(),
    )


def test_eval(features, kind, X_tr, y_tr, X_te, y_te):
    """Final check: fit on the whole training set, evaluate once on the test.

    Returns (test_f1_weighted, test_accuracy).
    """
    pipe = make_pipeline_for(features, kind)
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)

    return (
        f1_score(y_te, pred, average="weighted"),
        accuracy_score(y_te, pred),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # Load + preprocess (train for CV; external test for the final check)
    # =========================

    print("Loading and preprocessing data ...")

    def prepare(df):
        df = fix_dataframe_encoding(df)
        df = add_clean_text_column(
            df, source_column=TEXT_COLUMN, target_column=CLEAN_TEXT_COLUMN
        )
        df = add_clean_text_column(
            df,
            source_column=TEXT_COLUMN,
            target_column=TRANSFORMER_TEXT_COLUMN,
            cleaner=clean_text_light,
        )
        return df

    train_df = prepare(load_excel_dataset(TRAIN_FILE))
    test_df = prepare(load_csv_dataset(TEST_FILE))

    y = train_df[LABEL_COLUMN].to_numpy()
    X_text = train_df[CLEAN_TEXT_COLUMN].to_numpy(dtype=object)

    y_test = test_df[LABEL_COLUMN].to_numpy()
    X_test_text = test_df[CLEAN_TEXT_COLUMN].to_numpy(dtype=object)

    print(f"  training examples: {len(y)}  |  external test: {len(y_test)}")

    # =========================
    # BETO embeddings (one-shot, frozen encoder -> no leakage)
    # =========================

    print(f"Encoding BETO embeddings with {BETO_MODEL_NAME} ...")

    embedder = build_embedder(
        model_name=BETO_MODEL_NAME,
        max_length=BETO_MAX_LENGTH,
        batch_size=BETO_BATCH_SIZE,
    )

    X_beto = embedder.transform(train_df[TRANSFORMER_TEXT_COLUMN])
    X_test_beto = embedder.transform(test_df[TRANSFORMER_TEXT_COLUMN])

    print(f"  BETO shape: train {X_beto.shape}, test {X_test_beto.shape}")

    def get_X(features):
        return X_text if features == "TF-IDF" else X_beto

    def get_X_test(features):
        return X_test_text if features == "TF-IDF" else X_test_beto

    # =========================
    # 1) Learning curves (CV F1, train vs validation)
    # =========================

    print("\n=== 1) Learning curves (5-fold CV) ===")

    lc_data = []

    for name, features, kind in MODELS:
        print(f"  {name} ...")

        sizes, train_scores, val_scores = learning_curve(
            make_pipeline_for(features, kind),
            get_X(features),
            y,
            train_sizes=TRAIN_FRACTIONS,
            cv=CV,
            scoring="f1_weighted",
        )

        lc_data.append({
            "name": name,
            "sizes": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        })

    # Shared, normalized y-scale so the six curves are comparable.
    lc_low = min(
        float((d["val_mean"] - d["val_std"]).min()) for d in lc_data
    )
    lc_ylim = (lc_low - 0.03, 1.02)

    for d in lc_data:
        path = OUTPUT_DIR / f"learning_curve_{slug(d['name'])}.png"
        save_learning_curve_cv(
            d["sizes"], d["train_mean"], d["train_std"],
            d["val_mean"], d["val_std"], path, d["name"], ylim=lc_ylim,
        )
        print(f"  -> {path}")

    # =========================
    # 2) + 3) Overfitting figure and main comparison
    # =========================

    print("\n=== 2)/3) CV scores per model ===")

    names = []
    train_f1 = []
    cv_f1 = []
    cv_sd = []
    test_f1 = []
    test_acc = []

    for name, features, kind in MODELS:
        tr, cv, sd = cv_scores(features, kind, get_X(features), y)

        # Final check: train on the FULL training set, evaluate once on test.
        te_f1, te_acc = test_eval(
            features, kind, get_X(features), y, get_X_test(features), y_test
        )

        names.append(name)
        train_f1.append(tr)
        cv_f1.append(cv)
        cv_sd.append(sd)
        test_f1.append(te_f1)
        test_acc.append(te_acc)

        print(
            f"  {name:<24} train={tr:.3f}  CV={cv:.3f}±{sd:.3f}  "
            f"gap={tr - cv:.3f}  test={te_f1:.3f}"
        )

    # 2) Overfitting: grouped train vs CV bars + gap annotation.
    syn_low = max(0.0, min(cv_f1) - 0.06)
    save_overfit_synthesis(
        names, train_f1, cv_f1,
        OUTPUT_DIR / "overfit_synthesis.png",
        title=(
            "Sobreajuste — F1 entrenamiento vs. validación cruzada (5-fold)"
        ),
        ylim=(syn_low, 1.04),
    )
    print(f"\n-> {OUTPUT_DIR / 'overfit_synthesis.png'}")

    save_overfit_gap_bar(
        names, [t - c for t, c in zip(train_f1, cv_f1)],
        OUTPUT_DIR / "overfit_gap.png",
        title="Gap de sobreajuste (F1 entrenamiento − F1 validación cruzada, 5-fold)",
    )
    print(f"-> {OUTPUT_DIR / 'overfit_gap.png'}")

    # 3) Main comparison: mean CV F1 (sorted) with std error bars.
    cmp_low = max(0.0, min(np.array(cv_f1) - np.array(cv_sd)) - 0.03)
    save_model_comparison(
        names, cv_f1, cv_sd,
        OUTPUT_DIR / "model_comparison.png",
        title="Comparación de modelos — F1 validación cruzada (5-fold)",
        ylim=(cmp_low, min(1.0, float(max(np.array(cv_f1) + np.array(cv_sd))) + 0.03)),
    )
    print(f"-> {OUTPUT_DIR / 'model_comparison.png'}")

    # =========================
    # Final external-test evaluation: CV (selection) vs test (confirmation)
    # =========================

    print("\n=== Final external-test evaluation ===")
    for name, te_f1, te_acc in zip(names, test_f1, test_acc):
        print(f"  {name:<24} test_f1={te_f1:.3f}  test_acc={te_acc:.3f}")

    cvt_vals = np.concatenate([
        np.array(cv_f1) - np.array(cv_sd),
        np.array(test_f1),
    ])
    cvt_low = max(0.0, float(cvt_vals.min()) - 0.03)

    save_cv_vs_test(
        names, cv_f1, cv_sd, test_f1,
        OUTPUT_DIR / "cv_vs_test.png",
        title=(
            "Validación cruzada (selección) vs. test externo (confirmación) "
            "— modelos clásicos"
        ),
        ylim=(cvt_low, 1.02),
    )
    print(f"-> {OUTPUT_DIR / 'cv_vs_test.png'}")

    # =========================
    # Complementary: validation curves over the complexity knob
    # =========================

    print("\n=== Validation curves (complementary) ===")

    val_curves = []

    for name, features, kind in MODELS:
        param_name, values, xlabel, log_x, tick_labels = sweep_config(kind)
        print(f"  sweep {name}  ({param_name}) ...")

        tr_m, tr_s, cv_m, cv_s = sweep_overfitting(
            features, kind, get_X(features), y, param_name, values
        )

        x_vals = (
            np.arange(len(values)) if kind == "rf"
            else np.array(values, dtype=float)
        )

        val_curves.append({
            "name": name, "x_vals": x_vals,
            "train_mean": tr_m, "train_std": tr_s,
            "cv_mean": cv_m, "cv_std": cv_s,
            "xlabel": xlabel, "log_x": log_x, "tick_labels": tick_labels,
            "best_idx": int(np.argmax(cv_m)),
        })

    vc_scores = np.concatenate(
        [c["cv_mean"] for c in val_curves] + [c["train_mean"] for c in val_curves]
    )
    vc_ylim = (float(vc_scores.min()) - 0.03, 1.02)

    for c in val_curves:
        path = OUTPUT_DIR / f"validation_curve_{slug(c['name'])}.png"
        save_validation_curve(
            c["x_vals"], c["train_mean"], c["train_std"],
            c["cv_mean"], c["cv_std"], path, c["name"],
            xlabel=c["xlabel"], log_x=c["log_x"],
            x_tick_labels=c["tick_labels"], best_idx=c["best_idx"],
            ylim=vc_ylim,
        )
        print(f"  -> {path}")

    # =========================
    # Persist the numbers
    # =========================

    summary = [
        {
            "name": names[i],
            "cv_f1_mean": float(cv_f1[i]),
            "cv_f1_std": float(cv_sd[i]),
            "train_f1": float(train_f1[i]),
            "overfit_gap": float(train_f1[i] - cv_f1[i]),
            "test_f1": float(test_f1[i]),
            "test_accuracy": float(test_acc[i]),
        }
        for i in range(len(names))
    ]
    summary.sort(key=lambda r: r["cv_f1_mean"], reverse=True)

    summary_path = OUTPUT_DIR / "classical_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n-> {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
