"""Linear SVM (LinearSVC) classifier for the TF-IDF / BETO grid."""

from sklearn.svm import LinearSVC

from config import RANDOM_STATE, SVM_C, SVM_MAX_ITER


def build_model(C=SVM_C):
    """Build a balanced LinearSVC. ``C`` is tuned per representation by CV."""
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
    )
