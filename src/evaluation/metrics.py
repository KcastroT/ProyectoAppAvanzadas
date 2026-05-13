from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}\n")

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return y_pred