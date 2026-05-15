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

# config.py
from config import (
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_STATE,
    CLEAN_FILE_TEST,
    TEST_FILE,
    TEST_FIXED_FILE,
    TEXT_COLUMN,
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
    fix_dataframe_encoding,
)

# evaluation/metrics.py
from evaluation.metrics import (
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

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from models.svm_model import build_model




from sklearn.metrics import (
    accuracy_score,
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
    # Prepare ML data
    # =========================

    X = train_df[CLEAN_TEXT_COLUMN]

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

    X_test = test_df[CLEAN_TEXT_COLUMN]

    y_test = test_df[LABEL_COLUMN]

    # =========================
    # TF-IDF
    # =========================

    vectorizer = build_vectorizer()

    

    X_train_tfidf = vectorizer.fit_transform(
        X_train,
    )

    X_validation_tfidf = vectorizer.transform(
        X_validation,
    )

    X_test_tfidf = vectorizer.transform(
        X_test,
    )

    
    # =========================
    # Model
    # =========================

    model = build_model()

    model = train_model(
        model,
        X_train_tfidf,
        y_train,
    )
    
    # =========================
    # K-Fold Cross Validation
    # =========================

    cv_scores = cross_val_score(
        model,
        X_train_tfidf,
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
    # Train Evaluation
    # =========================

    train_predictions = model.predict(
        X_train_tfidf,
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    train_f1 = f1_score(
        y_train,
        train_predictions,
        average="weighted",
    )

    

    # =========================
    # Validation Evaluation
    # =========================

    print("\n=== Validation Evaluation ===")

    evaluate_model(
        model,
        X_validation_tfidf,
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
        X_test_tfidf,
        y_test,
    )
    
    results_df = build_results_dataframe(
        X_test,
        y_test,
        y_pred,
    )
    
    correct = get_correct_predictions(results_df)

    incorrect = get_incorrect_predictions(results_df)
    
    print("\n=== Correct Predictions ===")
    print(correct.sample(10))

    print("\n=== Incorrect Predictions ===")
    print(incorrect.sample(10))


if __name__ == "__main__":
    main()