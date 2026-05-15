from pathlib import Path

# =========================
# File Paths
# =========================

TRAIN_FILE = Path("../data/raw/data_train.xlsx")
TEST_FILE = Path("../data/raw/data_test_fold1(in).csv")

TRAIN_FIXED_FILE = Path("../data/processed/train_fixed_v2md.csv")
TEST_FIXED_FILE = Path("../data/processed/test_fixed_v2md.csv")

CLEAN_FILE_TRAIN = Path("../data/processed/train_cleaned_v2md.csv")
CLEAN_FILE_TEST = Path("../data/processed/test_cleaned_v2md.csv")

# =========================
# Dataset Columns
# =========================

TEXT_COLUMN = "tweet_text"
CLEAN_TEXT_COLUMN = "tweet_text_clean"
LABEL_COLUMN = "class"

VALIDATION_SIZE = 0.2

RANDOM_STATE = 42