from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(model, X, y):
    """Return predictions and core metrics as a dict (no printing).

    Handles both ``decision_function`` (SVM family) and ``predict_proba``
    (tree / NB / linear-model family) for AUC.
    """
    y_pred = model.predict(X)

    positive_class = model.classes_[1]

    y_binary = (y == positive_class).astype(int)

    auc = None

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        auc = roc_auc_score(y_binary, scores)

    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        auc = roc_auc_score(y_binary, probabilities[:, 1])

    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, average="weighted"),
        "auc": auc,
    }


def evaluate_model(model, X_test, y_test):
    """Compute metrics and print a verbose report (for single-model runs)."""
    metrics = compute_metrics(model, X_test, y_test)

    y_pred = metrics["y_pred"]

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    if metrics["auc"] is not None:
        print(f"AUC: {metrics['auc']:.4f}")

    print()

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return y_pred
