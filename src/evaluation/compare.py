"""Compare multiple classifiers fed the same feature representation."""

from sklearn.model_selection import cross_val_score

from evaluation.metrics import compute_metrics, evaluate_model


def compare_classifiers(
    classifiers,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    cv=5,
):
    """Fit each classifier, run K-fold CV on train, evaluate on val + test.

    Args:
        classifiers: dict mapping a display name to an unfitted estimator.
        X_train/y_train: training features and labels.
        X_validation/y_validation: held-out validation set.
        X_test/y_test: external test set.
        cv: number of folds for cross-validation on the training set.

    Returns:
        List of per-classifier result dicts (name, cv_mean, cv_std, val, test).
    """
    results = []

    for name, classifier in classifiers.items():
        print(f"\n--- Training {name} ---")

        classifier.fit(X_train, y_train)

        cv_scores = cross_val_score(
            classifier,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
        )

        val_metrics = compute_metrics(
            classifier,
            X_validation,
            y_validation,
        )

        test_metrics = compute_metrics(
            classifier,
            X_test,
            y_test,
        )

        results.append({
            "name": name,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "val": val_metrics,
            "test": test_metrics,
        })

        print(
            f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  |  "
            f"Val F1: {val_metrics['f1']:.4f}  |  "
            f"Test F1: {test_metrics['f1']:.4f}"
        )

    return results


def run_full_evaluation(
    name,
    classifier,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    cv=5,
):
    """Fit, cross-validate, and verbose-evaluate on validation + test.

    Mirrors the original single-model output (CV folds, classification
    report, confusion matrix) for each cell of a comparison grid, and
    returns a metrics dict for the summary table.
    """
    banner = "=" * 70

    print()
    print(banner)
    print(f"  {name}")
    print(banner)

    classifier.fit(X_train, y_train)

    # =========================
    # K-Fold Cross Validation
    # =========================

    cv_scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
    )

    print(f"\n=== {cv}-Fold Cross Validation ===")

    for idx, score in enumerate(cv_scores, start=1):
        print(f"Fold {idx}: {score:.4f}")

    print(f"\nAverage CV F1-score: {cv_scores.mean():.4f}")
    print(f"CV Standard Deviation: {cv_scores.std():.4f}")

    # =========================
    # Validation Evaluation
    # =========================

    print("\n=== Validation Evaluation ===")

    evaluate_model(classifier, X_validation, y_validation)

    val_metrics = compute_metrics(classifier, X_validation, y_validation)

    # =========================
    # External Test Evaluation
    # =========================

    print("\n=== External Test Evaluation ===")

    evaluate_model(classifier, X_test, y_test)

    test_metrics = compute_metrics(classifier, X_test, y_test)

    return {
        "name": name,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "val": val_metrics,
        "test": test_metrics,
    }


def print_comparison_grid(results, title="Comparison Grid"):
    """Print a wide grid: each row is (features, classifier) with all metrics."""
    width = 124

    print()
    print("=" * width)
    print(title)
    print("=" * width)

    header = (
        f"{'Features':<10} {'Classifier':<18} {'CV F1 (mean±std)':<22} "
        f"{'Val Acc':<10} {'Val F1':<10} {'Val AUC':<10} "
        f"{'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10}"
    )

    print(header)
    print("-" * width)

    for r in results:
        cv_str = f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"

        val = r["val"]
        test = r["test"]

        print(
            f"{r['features']:<10} {r['classifier']:<18} {cv_str:<22} "
            f"{val['accuracy']:<10.4f} {val['f1']:<10.4f} {val['auc']:<10.4f} "
            f"{test['accuracy']:<10.4f} {test['f1']:<10.4f} {test['auc']:<10.4f}"
        )

    print("=" * width)


def print_llm_comparison(results, title="LLM Comparison"):
    """Print a compact table comparing LLMs on validation + test.

    LLMs emit a label (not a score) and we skip CV for them, so this table
    omits CV / AUC columns. Each result dict needs ``name`` plus ``val`` and
    ``test`` metric dicts (with ``accuracy`` and ``f1``).
    """
    width = 74

    print()
    print("=" * width)
    print(title)
    print("=" * width)

    header = (
        f"{'Model':<22} {'Val Acc':<10} {'Val F1':<10} "
        f"{'Test Acc':<10} {'Test F1':<10}"
    )

    print(header)
    print("-" * width)

    for r in results:
        val = r["val"]
        test = r["test"]

        print(
            f"{r['name']:<22} "
            f"{val['accuracy']:<10.4f} {val['f1']:<10.4f} "
            f"{test['accuracy']:<10.4f} {test['f1']:<10.4f}"
        )

    print("=" * width)


def print_comparison_table(results, title="Classifier Comparison"):
    """Print a side-by-side comparison table of metrics."""
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)

    header = (
        f"{'Model':<24} {'CV F1 (mean±std)':<22} "
        f"{'Val Acc':<10} {'Val F1':<10} {'Val AUC':<10} "
        f"{'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10}"
    )

    print(header)
    print("-" * 104)

    for r in results:
        cv_str = f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"

        val = r["val"]
        test = r["test"]

        print(
            f"{r['name']:<24} {cv_str:<22} "
            f"{val['accuracy']:<10.4f} {val['f1']:<10.4f} {val['auc']:<10.4f} "
            f"{test['accuracy']:<10.4f} {test['f1']:<10.4f} {test['auc']:<10.4f}"
        )

    print("=" * 104)
