import re
import nltk
from typing import Dict

import ftfy
import pandas as pd
from nltk.stem.snowball import SnowballStemmer


SLANG_MAP: Dict[str, str] = {
    # =========================
    # Abreviaturas comunes
    # =========================
    "q": "que",
    "k": "que",
    "xq": "porque",
    "pq": "porque",
    "tqm": "te quiero mucho",
    "xfa": "por favor",
    "pls": "please",
    "u": "you",
    "bn": "bien",
    "ntp": "no te preocupes",
    "alv": "a la verga",
    "vrg": "verga",
    "tmb": "tambien",
    "sip": "si",
    "nop": "no",
    "holi": "hola",

    # =========================
    # Expresiones emocionales
    # =========================
    ":((": "triste",
    ":(": "triste",
    ":')": "feliz",
    "xd": "riendo",
    "xddd": "riendo",
    "jajaja": "riendo",
    "jeje": "riendo",
    "lmao": "riendo mucho",

    # =========================
    # TCA / dieta / fitness
    # =========================
    "thinspo": "inspiracion delgada",
    "thinspiration": "inspiracion delgada",
    "meanspo": "inspiracion agresiva",
    "fatspo": "miedo a gordura",
    "proana": "pro anorexia",
    "promia": "pro bulimia",
    "edtwt": "eating disorder twitter",
    "ana": "anorexia",
    "mia": "bulimia",

    
    # =========================
    # Variantes ortográficas
    # =========================
    "qiero": "quiero",
    "kiero": "quiero",
    "komer": "comer",
    "vomite": "vomitar",
    "vomitandooo": "vomitando",
}


# =========================
# Stemmer
# =========================

stemmer = SnowballStemmer("spanish")

# =========================
# Encoding
# =========================

def fix_encoding(text: str) -> str:
    """
    Fix encoding issues in a given text using ftfy.

    Args:
        text (str): The input text to fix.

    Returns:
        str: The text with encoding issues fixed.
    """
    if isinstance(text, str):
        return ftfy.fix_text(text)

    return text

def fix_dataframe_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix encoding issues for all object columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A copy of the DataFrame with fixed encoding.
    """
    df_copy = df.copy()

    for col in df_copy.select_dtypes(include="object").columns:
        df_copy[col] = df_copy[col].apply(fix_encoding)

    return df_copy

# =========================
# Text Cleaning Helpers
# =========================

def remove_urls(text: str) -> str:
    """
    Remove URLs from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without URLs.
    """
    return re.sub(r"http\S+|www\S+", "", text)

def remove_mentions(text: str) -> str:
    """
    Remove mentions (e.g., @username) from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without mentions.
    """
    return re.sub(r"@\w+", "", text)

def normalize_hashtags(text: str) -> str:
    """
    Normalize hashtags by removing the '#' symbol.

    Args:
        text (str): The input text.

    Returns:
        str: The text with normalized hashtags.
    """
    return re.sub(r"#(\w+)", r"\1", text)

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the given text by replacing multiple spaces with a single space.

    Args:
        text (str): The input text.

    Returns:
        str: The text with normalized whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()

def apply_stemming(text: str) -> str:
    """
    Apply stemming to the words in the given text using Snowball Stemmer.

    Args:
        text (str): The input text.

    Returns:
        str: The text with stemmed words.
    """
    words = text.split()

    stemmed_words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(stemmed_words)

def expand_slang(text: str, slang_map: Dict[str, str]) -> str:
    """
    Replace slang words in the text with their expanded forms using a slang map.

    Args:
        text (str): The input text.
        slang_map (Dict[str, str]): A dictionary mapping slang words to their expanded forms.

    Returns:
        str: The text with slang words expanded.
    """
    words = text.split()

    normalized_words = []

    for word in words:
        cleaned_word = re.sub(r"[^\w]", "", word)

        replacement = slang_map.get(cleaned_word, word)

        normalized_words.append(replacement)

    return " ".join(normalized_words)

# =========================
# Main Cleaning Pipeline
# =========================

def clean_text(text: str) -> str:
    """
    Apply a series of NLP preprocessing steps to clean the given text.

    Steps include:
        - Lowercasing
        - Removing URLs
        - Removing mentions
        - Normalizing hashtags
        - Normalizing whitespace
        - Expanding slang
        - Applying stemming

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return text

    text = text.lower()

    preprocessing_steps = [
        remove_urls,
        remove_mentions,
        normalize_hashtags,
        normalize_whitespace,
    ]

    for step in preprocessing_steps:
        text = step(text)

    text = expand_slang(text, SLANG_MAP)

    #text = apply_stemming(text)

    return text

def add_clean_text_column(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with cleaned text.

    Args:
        df (pd.DataFrame): The input DataFrame.
        source_column (str): The name of the column containing the original text.
        target_column (str): The name of the new column to store the cleaned text.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the new cleaned text column.
    """
    df_copy = df.copy()

    df_copy[target_column] = df_copy[source_column].apply(clean_text)

    return df_copy