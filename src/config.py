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
TRANSFORMER_TEXT_COLUMN = "tweet_text_transformer"
LABEL_COLUMN = "class"

VALIDATION_SIZE = 0.2

RANDOM_STATE = 42

# =========================
# Feature Extraction
# =========================

# Which representation feeds the classifier:
#   "tfidf"         - bag-of-words + LinearSVC (single-model flow)
#   "beto"          - frozen BETO embeddings + {LinearSVC, RandomForest}
#   "grid"          - 2x2 grid: {TF-IDF, BETO} x {LinearSVC, RandomForest}
#   "beto_finetune" - BETO fine-tuned end-to-end with a classification head
FEATURE_METHOD = "grid"

# =========================
# BETO Transformer (Spanish BERT)
# =========================

BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BETO_MAX_LENGTH = 128
BETO_BATCH_SIZE = 32

# =========================
# BETO Fine-tuning
# =========================

BETO_FT_LEARNING_RATE = 2e-5
BETO_FT_EPOCHS = 3
BETO_FT_BATCH_SIZE = 16
BETO_FT_WEIGHT_DECAY = 0.01
BETO_FT_WARMUP_RATIO = 0.1
BETO_FT_OUTPUT_DIR = Path("../runs/beto_ft")

# =========================
# Classifier Hyperparameters
# =========================
# Centralized so the academic report can cite a single source of truth.

# LinearSVC (used after StandardScaler on BETO embeddings, or on TF-IDF)
SVM_C = 1.0
SVM_MAX_ITER = 2000

# Random Forest (used on BETO embeddings)
RF_N_ESTIMATORS = 300