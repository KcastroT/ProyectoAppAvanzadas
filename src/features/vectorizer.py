from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    """
    Create and configure a TF-IDF vectorizer for text feature extraction.

    Returns
    -------
    sklearn.feature_extraction.text.TfidfVectorizer
        Configured TF-IDF vectorizer using character-level n-grams.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def train_vectorizer(vectorizer, X_train, X_test):
    """
    Fit a TF-IDF vectorizer on training data and transform datasets.

    Parameters
    ----------
    vectorizer : TF-IDF vectorizer instance to fit and apply.
    X_train : Training text samples used to fit the vectorizer.
    X_test : gTest text samples to transform using the fitted vectorizer.

    Returns
    -------
    tuple
        A tuple containing:

        - ``X_train_tfidf``: transformed TF-IDF matrix for training data
        - ``X_test_tfidf``: transformed TF-IDF matrix for test data
    """
    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf