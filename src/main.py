from config import (
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_CLEAN_FILE,
    TEST_FILE,
    TEST_FIXED_FILE,
    TEXT_COLUMN,
    TRAIN_CLEAN_FILE,
    TRAIN_FILE,
    TRAIN_FIXED_FILE,
)

from data.loader import (
    load_csv_dataset,
    load_excel_dataset,
)

from data.preprocessing import (
    add_clean_text_column,
    fix_dataframe_encoding,
)

from data.saving import save_dataframe

from evaluation.metrics import evaluate_model

from features.vectorizer import (
    build_vectorizer,
    train_vectorizer,
)

from models.svm_model import build_model

from evaluation.predictions import (
    build_results_dataframe,
    get_correct_predictions,
    get_incorrect_predictions,
)

def train_model(model, X_train, y_train):
    """Train ML model."""
    model.fit(X_train, y_train)

    return model


def main():

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

    save_dataframe(train_df, TRAIN_CLEAN_FILE)

    save_dataframe(test_df, TEST_CLEAN_FILE)

    # =========================
    # Prepare ML data
    # =========================

    X_train = train_df[CLEAN_TEXT_COLUMN]

    y_train = train_df[LABEL_COLUMN]

    X_test = test_df[CLEAN_TEXT_COLUMN]

    y_test = test_df[LABEL_COLUMN]

    # =========================
    # TF-IDF
    # =========================

    vectorizer = build_vectorizer()

    X_train_tfidf, X_test_tfidf = train_vectorizer(
        vectorizer,
        X_train,
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
    # Evaluation
    # =========================

    print("\n=== Model Information ===")
    print(f"Model: {model.__class__.__name__}")

    y_pred = evaluate_model(
    model,
    X_test_tfidf,
    y_test
)
    
    results_df = build_results_dataframe(
    X_test,
    y_test,
    y_pred,
)
    
    correct = get_correct_predictions(results_df)

    incorrect = get_incorrect_predictions(results_df)

    print("\n=== Correct Predictions ===")
    print(correct.sample(5))

    print("\n=== Incorrect Predictions ===")
    print(incorrect.sample(5))


if __name__ == "__main__":
    main()