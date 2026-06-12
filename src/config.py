from pathlib import Path

# =========================
# File Paths
# =========================

TRAIN_FILE = Path("../data/raw/data_train.xlsx")
TEST_FILE = Path("../data/raw/data_test_combined.csv")

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
#   "tfidf"         - tf-idf + LinearSVC 
#   "beto"          - frozen BETO embeddings + {LinearSVC, RandomForest}
#   "grid"          - 2x2 grid: {TF-IDF, BETO} x {LinearSVC, RandomForest}
#   "beto_finetune" - BETO fine-tuned end-to-end with a classification head
#   "llm"           - zero-/few-shot classification with a local LLM (Ollama)
FEATURE_METHOD = "llm"

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


SVM_C = 0.1
SVM_C_BETO = 0.1
SVM_MAX_ITER = 2000

# Random Forest (used on BETO embeddings)
RF_N_ESTIMATORS = 300

# Logistic Regression (probabilistic linear baseline; gives AUC on every cell)
LR_C = 1.0
LR_MAX_ITER = 2000

# =========================
# LLM comparison
# =========================
# Zero-/few-shot classification baseline.
#
# All three LLMs run locally through Ollama (server must be running, models
# pulled). Free hosted APIs (Gemini, Hugging Face) were dropped because their
# free tiers cap well below the ~675 calls this evaluation needs.
#   ollama pull llama3.2:3b
#   ollama pull qwen2.5:3b
#   ollama pull gemma2:2b
LLM_LABELS = ("anorexia", "control")
LLM_N_FEW_SHOT = 4  # examples per class embedded in the prompt (0 = zero-shot)

# Models compared head-to-head in the "llm" flow.
#   backend: "ollama" (local)
#   model:   ollama tag
LLM_MODELS = [
    {"name": "Llama 3.2 3B", "backend": "ollama", "model": "llama3.2:latest"},
    {"name": "Qwen 2.5 3B", "backend": "ollama", "model": "qwen2.5:3b"},
    {"name": "Gemma 2 2B", "backend": "ollama", "model": "gemma2:2b"},
]