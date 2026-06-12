"""
Main entry point for the Machine Learning pipeline.

This module orchestrates the complete workflow for:
- loading datasets
- fixing encoding issues
- preprocessing tweet text
- vectorizing text using TF-IDF
- training a machine learning classifier
- evaluating model performance
- analyzing predictions

The current implementation uses:
- TF-IDF for feature extraction
- Linear SVM for classification
"""

import numpy as np

# config.py
from config import (
    BETO_BATCH_SIZE,
    BETO_FT_BATCH_SIZE,
    BETO_FT_EPOCHS,
    BETO_FT_LEARNING_RATE,
    BETO_FT_OUTPUT_DIR,
    BETO_FT_WARMUP_RATIO,
    BETO_FT_WEIGHT_DECAY,
    BETO_MAX_LENGTH,
    BETO_MODEL_NAME,
    CLEAN_TEXT_COLUMN,
    FEATURE_METHOD,
    LABEL_COLUMN,
    LLM_LABELS,
    LLM_MODELS,
    LLM_N_FEW_SHOT,
    RANDOM_STATE,
    SVM_C_BETO,
    CLEAN_FILE_TEST,
    TEST_FILE,
    TEST_FIXED_FILE,
    TEXT_COLUMN,
    TRANSFORMER_TEXT_COLUMN,
    CLEAN_FILE_TRAIN,
    TRAIN_FILE,
    TRAIN_FIXED_FILE,
    VALIDATION_SIZE,
)

# data/loading.py
from data.loader import (
    load_csv_dataset,
    load_excel_dataset,
)

# data/saving.py
from data.saving import save_dataframe

# preprocessing.py
from data.preprocessing import (
    add_clean_text_column,
    clean_text_light,
    fix_dataframe_encoding,
)

# evaluation/metrics.py
from evaluation.metrics import (
    compute_metrics,
    evaluate_model,
)

from evaluation.predictions import (
    build_results_dataframe,
    get_correct_predictions,
    get_incorrect_predictions,
)

# features/vectorizer.py
from features.vectorizer import (
    build_vectorizer
)

# features/embeddings.py
from features.embeddings import (
    build_embedder
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from models.svm_model import build_model as build_svm_model

from models.random_forest_model import (
    build_model as build_rf_model,
)

from models.logistic_regression_model import (
    build_model as build_lr_model,
)

from models.beto_finetune import (
    build_model as build_beto_finetune_model,
)

from models.llm import (
    build_model as build_llm_model,
)

from evaluation.compare import (
    compare_classifiers,
    print_comparison_grid,
    print_comparison_table,
    print_llm_comparison,
    print_metrics_summary,
    run_full_evaluation,
)




from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

def train_model(model, X_train, y_train):
    """Train a machine learning model using TF-IDF features.

    Args:
        model: Scikit-learn compatible classifier.
        X_train: Vectorized training features.
        y_train: Training labels.

    Returns:
        Trained machine learning model.
    """
    model.fit(X_train, y_train)

    return model


def main():
    """Execute the complete NLP classification pipeline.

    Workflow:
        1. Load training and testing datasets.
        2. Fix text encoding issues.
        3. Clean and preprocess tweet text.
        4. Save processed datasets.
        5. Extract TF-IDF features.
        6. Train the machine learning classifier.
        7. Evaluate model performance.
        8. Display correct and incorrect predictions.
    """

    # =========================
    # Load datasets
    # =========================

    train_df = load_excel_dataset(TRAIN_FILE)

    test_df = load_csv_dataset(TEST_FILE)

    # =========================
    # Fix encoding
    # =========================

    train_df = fix_dataframe_encoding(train_df)

    test_df = fix_dataframe_encoding(test_df)

    save_dataframe(train_df, TRAIN_FIXED_FILE)

    save_dataframe(test_df, TEST_FIXED_FILE)

    # =========================
    # Text cleaning
    # =========================

    train_df = add_clean_text_column(
        train_df,
        source_column=TEXT_COLUMN,
        target_column=CLEAN_TEXT_COLUMN,
    )

    test_df = add_clean_text_column(
        test_df,
        source_column=TEXT_COLUMN,
        target_column=CLEAN_TEXT_COLUMN,
    )

    save_dataframe(train_df, CLEAN_FILE_TRAIN)

    save_dataframe(test_df, CLEAN_FILE_TEST)

    # =========================
    # Transformer text (BETO)
    # =========================
    #
    # BETO is a *cased* subword model, so it works better on lightly cleaned
    # text (case and accents preserved, no stemming) than on the aggressively
    # normalized TF-IDF column.

    if FEATURE_METHOD in ("beto", "beto_finetune", "grid", "llm"):
        train_df = add_clean_text_column(
            train_df,
            source_column=TEXT_COLUMN,
            target_column=TRANSFORMER_TEXT_COLUMN,
            cleaner=clean_text_light,
        )

        test_df = add_clean_text_column(
            test_df,
            source_column=TEXT_COLUMN,
            target_column=TRANSFORMER_TEXT_COLUMN,
            cleaner=clean_text_light,
        )

    if FEATURE_METHOD == "grid":
        text_column = None  # grid mode pulls both columns directly below
    elif FEATURE_METHOD in ("beto", "beto_finetune", "llm"):
        text_column = TRANSFORMER_TEXT_COLUMN
    else:
        text_column = CLEAN_TEXT_COLUMN

    # =========================
    # Prepare ML data
    # =========================

    if FEATURE_METHOD == "grid":
        # Split the DataFrame so both text columns stay aligned to the same
        # train/validation indices.
        train_df_split, validation_df_split = train_test_split(
            train_df,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df[LABEL_COLUMN],
        )

        y_train = train_df_split[LABEL_COLUMN]

        y_validation = validation_df_split[LABEL_COLUMN]

        y_test = test_df[LABEL_COLUMN]
    else:
        X = train_df[text_column]

        y = train_df[LABEL_COLUMN]

        X_train, X_validation, y_train, y_validation = (
            train_test_split(
                X,
                y,
                test_size=VALIDATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        )

        X_test = test_df[text_column]

        y_test = test_df[LABEL_COLUMN]

    # =========================
    # Feature extraction
    # =========================

    print(f"\n=== Feature Method: {FEATURE_METHOD} ===")

    if FEATURE_METHOD == "grid":
        # =========================
        # Build TF-IDF features (one-shot)
        # =========================

        print("\n=== Building TF-IDF features ===")

        vectorizer = build_vectorizer()

        X_train_tfidf = vectorizer.fit_transform(
            train_df_split[CLEAN_TEXT_COLUMN]
        )

        X_validation_tfidf = vectorizer.transform(
            validation_df_split[CLEAN_TEXT_COLUMN]
        )

        X_test_tfidf = vectorizer.transform(
            test_df[CLEAN_TEXT_COLUMN]
        )

        print(
            f"TF-IDF vocabulary size: "
            f"{len(vectorizer.get_feature_names_out())}"
        )

        # =========================
        # Build BETO embeddings (one-shot)
        # =========================

        print("\n=== Building BETO embeddings ===")

        embedder = build_embedder(
            model_name=BETO_MODEL_NAME,
            max_length=BETO_MAX_LENGTH,
            batch_size=BETO_BATCH_SIZE,
        )

        print(f"Encoding with {BETO_MODEL_NAME} on {embedder.device} ...")

        X_train_beto = embedder.transform(
            train_df_split[TRANSFORMER_TEXT_COLUMN]
        )

        X_validation_beto = embedder.transform(
            validation_df_split[TRANSFORMER_TEXT_COLUMN]
        )

        X_test_beto = embedder.transform(
            test_df[TRANSFORMER_TEXT_COLUMN]
        )

        print(
            f"BETO embedding shape: {X_train_beto.shape} "
            f"(samples x hidden_size)"
        )

        # =========================
        # 2x3 Grid: features × classifier
        # =========================
        # Scaling rules:
        #   TF-IDF + LinearSVC / LogReg: sparse non-negative input, no scaler.
        #   TF-IDF + RandomForest: scale-invariant.
        #   BETO   + LinearSVC / LogReg: dense embeddings -> StandardScaler.
        #   BETO   + RandomForest: scale-invariant.

        combinations = [
            (
                "TF-IDF",
                "LinearSVC",
                build_svm_model(),
                X_train_tfidf,
                X_validation_tfidf,
                X_test_tfidf,
            ),
            (
                "TF-IDF",
                "RandomForest",
                build_rf_model(),
                X_train_tfidf,
                X_validation_tfidf,
                X_test_tfidf,
            ),
            (
                "TF-IDF",
                "LogisticReg",
                build_lr_model(),
                X_train_tfidf,
                X_validation_tfidf,
                X_test_tfidf,
            ),
            (
                "BETO",
                "LinearSVC",
                make_pipeline(StandardScaler(), build_svm_model(C=SVM_C_BETO)),
                X_train_beto,
                X_validation_beto,
                X_test_beto,
            ),
            (
                "BETO",
                "RandomForest",
                build_rf_model(),
                X_train_beto,
                X_validation_beto,
                X_test_beto,
            ),
            (
                "BETO",
                "LogisticReg",
                make_pipeline(StandardScaler(), build_lr_model()),
                X_train_beto,
                X_validation_beto,
                X_test_beto,
            ),
        ]

        grid_results = []

        for features, clf_name, classifier, X_tr, X_v, X_te in combinations:
            result = run_full_evaluation(
                name=f"{features} + {clf_name}",
                classifier=classifier,
                X_train=X_tr,
                y_train=y_train,
                X_validation=X_v,
                y_validation=y_validation,
                X_test=X_te,
                y_test=y_test,
            )

            result["features"] = features

            result["classifier"] = clf_name

            grid_results.append(result)

        print_comparison_grid(
            grid_results,
            title=(
                "2x3 Grid: (TF-IDF, BETO) x "
                "(LinearSVC, RandomForest, LogisticReg)"
            ),
        )

        # Compact 5-metric tables (accuracy, precision, recall, F1, AUC).
        print_metrics_summary(
            grid_results,
            split="val",
            title="Validation metrics (weighted)",
        )
        print_metrics_summary(
            grid_results,
            split="test",
            title="External test metrics (weighted)",
        )

        return

    if FEATURE_METHOD == "beto_finetune":
        X_train_features = X_train

        X_validation_features = X_validation

        X_test_features = X_test

        print(
            f"Fine-tuning {BETO_MODEL_NAME} "
            f"({BETO_FT_EPOCHS} epochs, lr={BETO_FT_LEARNING_RATE}, "
            f"batch={BETO_FT_BATCH_SIZE}, seed={RANDOM_STATE}) ..."
        )

        model = build_beto_finetune_model(
            model_name=BETO_MODEL_NAME,
            max_length=BETO_MAX_LENGTH,
            batch_size=BETO_FT_BATCH_SIZE,
            epochs=BETO_FT_EPOCHS,
            learning_rate=BETO_FT_LEARNING_RATE,
            weight_decay=BETO_FT_WEIGHT_DECAY,
            warmup_ratio=BETO_FT_WARMUP_RATIO,
            seed=RANDOM_STATE,
            output_dir=str(BETO_FT_OUTPUT_DIR),
        )
    elif FEATURE_METHOD == "beto":
        embedder = build_embedder(
            model_name=BETO_MODEL_NAME,
            max_length=BETO_MAX_LENGTH,
            batch_size=BETO_BATCH_SIZE,
        )

        print(f"Encoding with {BETO_MODEL_NAME} on {embedder.device} ...")

        X_train_features = embedder.transform(X_train)

        X_validation_features = embedder.transform(X_validation)

        X_test_features = embedder.transform(X_test)

        print(
            f"Embedding shape: {X_train_features.shape} "
            f"(samples x hidden_size)"
        )

        # Compare classical classifiers on the same BETO embeddings.
        # LinearSVC is sensitive to feature scale, so it goes through a
        # StandardScaler; Random Forest is scale-invariant, so it doesn't.
        classifiers = {
            "LinearSVC": make_pipeline(
                StandardScaler(),
                build_svm_model(),
            ),
            "RandomForestClassifier": build_rf_model(),
        }

        results = compare_classifiers(
            classifiers,
            X_train_features,
            y_train,
            X_validation_features,
            y_validation,
            X_test_features,
            y_test,
        )

        print_comparison_table(
            results,
            title="Classifier Comparison (frozen BETO embeddings)",
        )

        return
    elif FEATURE_METHOD == "llm":
        # Compare LLMs head-to-head with zero-/few-shot prompting: three local
        # models (Llama, Qwen, Gemma) served via Ollama. Each prediction is
        # an LLM call, so we skip K-fold CV (it would mean thousands of calls)
        # and evaluate directly on validation + test. The models also report a
        # probability (prob_anorexia), so AUC is available via predict_proba.
        def build_llm(spec):
            """Build the right classifier for a model spec by backend."""
            backend = spec["backend"]

            if backend == "ollama":
                return build_llm_model(
                    model_name=spec["model"],
                    labels=LLM_LABELS,
                    n_few_shot=LLM_N_FEW_SHOT,
                )

            raise ValueError(f"Unknown LLM backend: {backend!r}")

        shot_kind = (
            f"{LLM_N_FEW_SHOT}-shot per class"
            if LLM_N_FEW_SHOT > 0
            else "zero-shot"
        )

        def report_split(split_name, y_true, metrics):
            """Print one split's metrics + report from a single prediction pass.

            ``metrics`` comes from ``compute_metrics`` (which already ran
            ``predict`` once), so we reuse its ``y_pred`` instead of querying
            the LLM again.
            """
            y_pred = metrics["y_pred"]

            print(f"\n=== {split_name} Evaluation ===")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")

            if metrics.get("auc") is not None:
                print(f"AUC: {metrics['auc']:.4f}")

            print("\n=== Classification Report ===")
            print(classification_report(y_true, y_pred))

            print("=== Confusion Matrix ===")
            print(confusion_matrix(y_true, y_pred))

        llm_results = []

        for spec in LLM_MODELS:
            display_name = spec["name"]

            print()
            print("=" * 70)
            print(
                f"  {display_name}  "
                f"({spec['backend']}:{spec['model']}, {shot_kind})"
            )
            print("=" * 70)

            model = build_llm(spec)

            # "Training" only records labels and samples few-shot examples.
            model.fit(X_train, y_train)

            # One prediction pass per split (LLM calls are expensive).
            val_metrics = compute_metrics(model, X_validation, y_validation)

            test_metrics = compute_metrics(model, X_test, y_test)

            report_split("Validation", y_validation, val_metrics)

            report_split("External Test", y_test, test_metrics)

            llm_results.append({
                "name": display_name,
                "val": val_metrics,
                "test": test_metrics,
            })

        print_llm_comparison(
            llm_results,
            title=f"LLM Comparison (Ollama, {shot_kind})",
        )

        return
    else:
        vectorizer = build_vectorizer()

        X_train_features = vectorizer.fit_transform(X_train)

        X_validation_features = vectorizer.transform(X_validation)

        X_test_features = vectorizer.transform(X_test)

        feature_names = vectorizer.get_feature_names_out()

        print("\n=== TF-IDF Vocabulary ===")
        print(f"Vocabulary size: {len(feature_names)}")

        sample_size = min(20, len(feature_names))

        rng = np.random.default_rng(RANDOM_STATE)

        sample_indices = rng.choice(
            len(feature_names),
            size=sample_size,
            replace=False,
        )

        print(f"\nRandom sample of {sample_size} tokens:")

        for idx in sample_indices:
            print(f"  - {feature_names[idx]}")

        model = build_svm_model()

    # =========================
    # Model
    # =========================

    model = train_model(
        model,
        X_train_features,
        y_train,
    )
    
    # =========================
    # K-Fold Cross Validation
    # =========================

    if FEATURE_METHOD == "beto_finetune":
        print("\n=== 5-Fold Cross Validation ===")
        print(
            "(skipped for beto_finetune: refitting a transformer per fold "
            "is expensive — rely on validation/test metrics, and re-run "
            "with different RANDOM_STATE values to estimate variance)"
        )
    else:
        cv_scores = cross_val_score(
            model,
            X_train_features,
            y_train,
            cv=5,
            scoring="f1_weighted",
        )

        print("\n=== 5-Fold Cross Validation ===")

        for idx, score in enumerate(cv_scores, start=1):
            print(f"Fold {idx}: {score:.4f}")

        print(
            f"\nAverage CV F1-score: "
            f"{cv_scores.mean():.4f}"
        )

        print(
            f"CV Standard Deviation: "
            f"{cv_scores.std():.4f}"
        )


    # =========================
    # Validation Evaluation
    # =========================

    print("\n=== Validation Evaluation ===")

    evaluate_model(
        model,
        X_validation_features,
        y_validation,
    )

    # =========================
    # Evaluation
    # =========================

    print("\n=== Model Information ===")
    print(f"Model: {model.__class__.__name__}")

    print("\n=== External Test Evaluation ===")

    y_pred = evaluate_model(
        model,
        X_test_features,
        y_test,
    )
    
    results_df = build_results_dataframe(
        X_test,
        y_test,
        y_pred,
    )
    
    correct = get_correct_predictions(results_df)

    incorrect = get_incorrect_predictions(results_df)
    
    # print("\n=== Correct Predictions ===")
    # print(correct.sample(10))

    # print("\n=== Incorrect Predictions ===")
    # print(incorrect.sample(10))


if __name__ == "__main__":
    main()