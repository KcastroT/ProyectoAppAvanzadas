from sklearn.feature_extraction.text import TfidfVectorizer



def build_vectorizer():
    """Create TF-IDF vectorizer."""
    return TfidfVectorizer(
        analyzer="char_wb",
        max_features=2000,
        ngram_range=(2,5),
        min_df=3,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def train_vectorizer(vectorizer, X_train, X_test):
    """Fit and transform TF-IDF features."""
    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf