from sklearn.svm import LinearSVC

from config import RANDOM_STATE, SVM_C, SVM_MAX_ITER


def build_model():
    """Linear SVM tuned for dense BETO embeddings (also OK on TF-IDF).

    - C=1.0 is the canonical default; with StandardScaler-normalized BETO
      features, lower values over-regularize the 768-dim space.
    - class_weight='balanced' counters the mild class skew (~54/46).
    - max_iter is bumped above the default to avoid convergence warnings.
    - random_state fixes any stochastic tiebreaks for reproducibility.
    """
    return LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
    )
