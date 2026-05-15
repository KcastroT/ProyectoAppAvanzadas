from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # =========================
    # AUC Support for Multiple Models
    # =========================

    auc = None

    positive_class = model.classes_[1]

    y_test_binary = (
        y_test == positive_class
    ).astype(int)

    if hasattr(model, "decision_function"):

        decision_scores = model.decision_function(X_test)

        auc = roc_auc_score(
            y_test_binary,
            decision_scores,
        )

    elif hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        positive_scores = probabilities[:, 1]

        auc = roc_auc_score(
            y_test_binary,
            positive_scores,
        )

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    if auc is not None:
        print(f"AUC: {auc:.4f}")

    print()

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return y_pred