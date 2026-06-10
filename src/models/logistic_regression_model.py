from sklearn.linear_model import LogisticRegression

from config import LR_C, LR_MAX_ITER, RANDOM_STATE


def build_model():
    """Logistic Regression: probabilistic linear baseline.

    - Complements LinearSVC (max-margin) and RandomForest (ensemble) as the
      third classical classifier; works on both sparse TF-IDF and dense BETO.
    - Exposes predict_proba, so AUC is available on every grid cell.
    - class_weight='balanced' matches the SVM/RF setup for a fair comparison.
    - max_iter is bumped above the default to converge on the 768-dim BETO space.
    """
    return LogisticRegression(
        C=LR_C,
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )
