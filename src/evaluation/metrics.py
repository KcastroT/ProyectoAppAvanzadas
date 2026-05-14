from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_auc_custom(TP, FP, FN, TN):
    """
    Calcula AUC usando la fórmula especificada:
    AUC = (1 + TPR - FPR) / 2
    
    Donde:
    - TPR = TP / (TP + FN)  [True Positive Rate / Sensibilidad]
    - FPR = FP / (FP + TN)  [False Positive Rate]
    """
    if (TP + FN) == 0 or (FP + TN) == 0:
        return None
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    
    auc = (1 + TPR - FPR) / 2
    
    return auc


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # =========================
    # AUC usando fórmula especificada
    # =========================

    auc = None

    positive_class = model.classes_[1]

    y_test_binary = (
        y_test == positive_class
    ).astype(int)

    # Calcular matriz de confusión manualmente
    TP = sum((y_pred == 1) & (y_test_binary == 1))
    FP = sum((y_pred == 1) & (y_test_binary == 0))
    TN = sum((y_pred == 0) & (y_test_binary == 0))
    FN = sum((y_pred == 0) & (y_test_binary == 1))

    auc = calculate_auc_custom(TP, FP, FN, TN)

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