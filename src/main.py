from pathlib import Path
import re

import ftfy
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC


TRAIN_FILE = Path("../data/raw/data_train.xlsx")
TEST_FILE = Path("../data/raw/data_test_fold1(in).csv")

TRAIN_FIXED_FILE = Path("../data/processed/train_fixed_v2md.csv")
TEST_FIXED_FILE = Path("../data/processed/test_fixed_v2md.csv")

TRAIN_CLEAN_FILE = Path("../data/processed/train_cleaned_v2md.csv")
TEST_CLEAN_FILE = Path("../data/processed/test_cleaned_v2md.csv")

TEXT_COLUMN = "tweet_text"
CLEAN_TEXT_COLUMN = "tweet_text_clean"
LABEL_COLUMN = "class"


# =========================
# Data Loading
# =========================

def load_excel_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from an Excel file."""
    return pd.read_excel(file_path)


def load_csv_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


# =========================
# Encoding Cleaning
# =========================

def fix_encoding(text: str) -> str:
    """Fix encoding issues using ftfy."""
    if isinstance(text, str):
        return ftfy.fix_text(text)
    return text



def fix_dataframe_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply encoding fixes to all object columns."""
    df_copy = df.copy()

    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = df_copy[col].apply(fix_encoding)

    return df_copy


# =========================
# Text Cleaning
# =========================

def clean_text(text: str) -> str:
    """Apply NLP preprocessing."""
    if not isinstance(text, str):
        return text

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Preserve hashtags removing only '#'
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def add_clean_text_column(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Create a cleaned text column."""
    df_copy = df.copy()

    df_copy[target_column] = df_copy[source_column].apply(clean_text)

    return df_copy


# =========================
# Data Saving
# =========================

def save_dataframe(df: pd.DataFrame, file_path: Path) -> None:
    """Save dataframe as UTF-8 CSV."""
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


# =========================
# ML Pipeline
# =========================




def build_vectorizer() -> TfidfVectorizer:
    """Create TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )



def train_vectorizer(vectorizer, X_train, X_test):
    """Fit and transform TF-IDF features."""
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf



def build_model() -> LinearSVC:
    """Create Linear SVM model."""
    return LinearSVC(C=1.0)



def train_model(model, X_train, y_train):
    """Train the model."""
    model.fit(X_train, y_train)

    return model



def evaluate_model(model, X_test, y_test) -> None:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}\n")

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))


# =========================
# Main Pipeline
# =========================

def main():
    # Load datasets
    train_df = load_excel_dataset(TRAIN_FILE)
    test_df = load_csv_dataset(TEST_FILE)

    # Fix encoding issues
    train_df_fixed = fix_dataframe_encoding(train_df)
    test_df_fixed = fix_dataframe_encoding(test_df)

    save_dataframe(train_df_fixed, TRAIN_FIXED_FILE)
    save_dataframe(test_df_fixed, TEST_FIXED_FILE)

    # Apply text cleaning
    train_df_clean = add_clean_text_column(
        train_df_fixed,
        source_column=TEXT_COLUMN,
        target_column=CLEAN_TEXT_COLUMN,
    )

    test_df_clean = add_clean_text_column(
        test_df_fixed,
        source_column=TEXT_COLUMN,
        target_column=CLEAN_TEXT_COLUMN,
    )

    save_dataframe(train_df_clean, TRAIN_CLEAN_FILE)
    save_dataframe(test_df_clean, TEST_CLEAN_FILE)

    # Prepare ML data
    X_train = train_df_clean[CLEAN_TEXT_COLUMN]
    y_train = train_df_clean[LABEL_COLUMN]

    X_test = test_df_clean[CLEAN_TEXT_COLUMN]
    y_test = test_df_clean[LABEL_COLUMN]

    # TF-IDF
    vectorizer = build_vectorizer()
    X_train_tfidf, X_test_tfidf = train_vectorizer(
        vectorizer,
        X_train,
        X_test,
    )

    # Train SVM model
    model = build_model()
    model = train_model(model, X_train_tfidf, y_train)

    # Evaluate
    evaluate_model(model, X_test_tfidf, y_test)


if __name__ == "__main__":
    main()