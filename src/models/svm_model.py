from sklearn.svm import LinearSVC


def build_model():
    """Create Linear SVM model."""
    return LinearSVC(C=0.5, class_weight="balanced")