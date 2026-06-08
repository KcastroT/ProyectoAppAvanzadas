from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, RF_N_ESTIMATORS


def build_model():
    """Random Forest tuned for dense BETO embeddings.

    - n_estimators=300 reduces variance on the 768-dim embedding space;
      diminishing returns past ~300 trees.
    - class_weight='balanced' matches the SVM configuration so the two
      classifiers are compared on equal terms.
    - n_jobs=-1 parallelizes tree fitting and prediction across all cores.
    - max_features defaults to 'sqrt' (~28 features per split for 768 dims),
      which is appropriate for high-dimensional dense input.
    """
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
