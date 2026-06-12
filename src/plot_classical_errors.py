"""Generalization diagnostics for the six classical models.

Everything here is grounded in **stratified 5-fold cross-validation** over the
full training set, using **weighted F1** (not accuracy) to avoid being misled
by class imbalance. No single train/test split is used as a quality signal.

Deliverables (run from ``src/``: ``python plot_classical_errors.py``):

  1. Main comparison — mean CV F1 per model with std error bars, sorted
     best→worst. Answers "which model generalizes best to unseen data?".
  2. Overfitting figure — grouped train vs CV F1 bars with the gap annotated
     (green <0.05, orange 0.05-0.10, red >0.10), plus a gap bar.
  3. CV (selection) vs external-test (confirmation): F1 and ROC/AUC.

TF-IDF is vectorized inside the CV pipeline (refit on each fold, no leakage);
BETO embeddings come from a frozen, label-independent encoder, so they are
computed once and reused.

Figures are written to ``../reports/figures/``.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BETO_BATCH_SIZE,
    BETO_MAX_LENGTH,
    BETO_MODEL_NAME,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_STATE,
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
from evaluation.plots import (
    save_auc_cv_vs_test,
    save_cv_vs_test,
    save_model_comparison,
    save_overfit_gap_bar,
    save_overfit_synthesis,
    save_roc_cv_test,
)
from features.embeddings import build_embedder
from features.vectorizer import build_vectorizer
from models.logistic_regression_model import build_model as build_lr
from models.random_forest_model import build_model as build_rf
from models.svm_model import build_model as build_svm

OUTPUT_DIR = Path("../reports/figures")

# 5-fold stratified CV, shared by every diagnostic.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

MODELS = [
    ("TF-IDF + LinearSVC", "TF-IDF", "svm"),
    ("TF-IDF + LogisticReg", "TF-IDF", "lr"),
    ("TF-IDF + RandomForest", "TF-IDF", "rf"),
    ("BETO + LinearSVC", "BETO", "svm"),
    ("BETO + LogisticReg", "BETO", "lr"),
    ("BETO + RandomForest", "BETO", "rf"),
]


def base_estimator(kind, features):
    """Fresh base classifier, matching main.py's grid configuration.

    The SVM uses a representation-specific C (tuned by CV): the default
    ``SVM_C`` for TF-IDF, and the heavier-regularized ``SVM_C_BETO`` for the
    dense BETO embeddings.
    """
    if kind == "svm":
        return build_svm(C=SVM_C_BETO) if features == "BETO" else build_svm()
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

    steps.append(("clf", base_estimator(kind, features)))

    return Pipeline(steps)


def slug(name):
    """Filesystem-safe slug for a model name."""
    return name.lower().replace(" + ", "_").replace(" ", "").replace("-", "")


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


# Class we treat as "positive" for ROC/AUC (the condition to detect).
POS_LABEL = "anorexia"


def positive_scores(fitted, X):
    """Continuous score for POS_LABEL (decision_function or predict_proba).

    Higher = more likely POS_LABEL. Works for a fitted Pipeline; LinearSVC's
    decision_function is signed toward classes_[1], so flip it if POS_LABEL is
    classes_[0].
    """
    classes = list(fitted.classes_)

    if hasattr(fitted, "decision_function"):
        score = fitted.decision_function(X)
        return score if POS_LABEL == classes[1] else -score

    proba = fitted.predict_proba(X)
    return proba[:, classes.index(POS_LABEL)]


def roc_cv_and_test(features, kind, X_tr, y_tr, X_te, y_te):
    """Per-fold CV ROC (mean ±std) + external-test ROC for one model.

    Returns a dict with the interpolated CV ROC, its AUC mean/std, and the
    test ROC + AUC.
    """
    mean_fpr = np.linspace(0.0, 1.0, 100)

    tprs = []
    aucs = []

    for train_idx, val_idx in CV.split(X_tr, y_tr):
        pipe = make_pipeline_for(features, kind)
        pipe.fit(_take(X_tr, train_idx), y_tr[train_idx])

        scores = positive_scores(pipe, _take(X_tr, val_idx))
        fpr, tpr, _ = roc_curve(
            y_tr[val_idx], scores, pos_label=POS_LABEL
        )

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # External test: fit on the full training set, score once on test.
    pipe = make_pipeline_for(features, kind)
    pipe.fit(X_tr, y_tr)
    test_scores = positive_scores(pipe, X_te)
    test_fpr, test_tpr, _ = roc_curve(y_te, test_scores, pos_label=POS_LABEL)

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": np.std(tprs, axis=0),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "test_fpr": test_fpr,
        "test_tpr": test_tpr,
        "test_auc": float(auc(test_fpr, test_tpr)),
    }


def _take(X, idx):
    """Index rows of either a numpy array or an object array of texts."""
    return X[idx]


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
    # CV scores: overfitting figure and main comparison
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
    # ROC curves + AUC (validation via CV, and external test)
    # =========================

    print(f"\n=== ROC / AUC (positive class = {POS_LABEL}) ===")

    auc_cv_mean = []
    auc_cv_std = []
    auc_test = []

    for name, features, kind in MODELS:
        roc = roc_cv_and_test(
            features, kind, get_X(features), y, get_X_test(features), y_test
        )

        auc_cv_mean.append(roc["cv_auc_mean"])
        auc_cv_std.append(roc["cv_auc_std"])
        auc_test.append(roc["test_auc"])

        # The SVM (LinearSVC) cells use a CV-tuned C, so flag them as optimized.
        roc_title = f"{name} (Optimizado)" if kind == "svm" else name

        save_roc_cv_test(
            roc["mean_fpr"], roc["mean_tpr"], roc["std_tpr"],
            roc["cv_auc_mean"], roc["cv_auc_std"],
            roc["test_fpr"], roc["test_tpr"], roc["test_auc"],
            OUTPUT_DIR / f"roc_{slug(name)}.png", roc_title, pos_label=POS_LABEL,
        )

        print(
            f"  {name:<24} AUC_cv={roc['cv_auc_mean']:.3f}±"
            f"{roc['cv_auc_std']:.3f}  AUC_test={roc['test_auc']:.3f}"
        )

    auc_low = max(
        0.5,
        min(
            float(min(np.array(auc_cv_mean) - np.array(auc_cv_std))),
            float(min(auc_test)),
        ) - 0.03,
    )
    save_auc_cv_vs_test(
        names, auc_cv_mean, auc_cv_std, auc_test,
        OUTPUT_DIR / "auc_cv_vs_test.png",
        title="AUC — validación cruzada vs. test externo (modelos clásicos)",
        ylim=(auc_low, 1.005),
    )
    print(f"-> {OUTPUT_DIR / 'auc_cv_vs_test.png'}")

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
            "auc_cv_mean": float(auc_cv_mean[i]),
            "auc_cv_std": float(auc_cv_std[i]),
            "auc_test": float(auc_test[i]),
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
