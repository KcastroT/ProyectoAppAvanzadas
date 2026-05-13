from sklearn.linear_model import LogisticRegression


def build_model():
    """Create Logistic Regression model."""
    return LogisticRegression(max_iter=1000)