import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation.metrics import compute_metrics


# Two linearly separable classes (1-D feature) used across the metric tests.
# compute_metrics does an element-wise ``y == positive_class`` (as in the real
# pipeline, where y is a pandas Series / numpy array), so y must be array-like.
X = [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]]
Y = np.array(
    ["control", "control", "control", "anorexia", "anorexia", "anorexia"]
)


class TestMetrics(unittest.TestCase):

    def test_model_has_predict(self):

        self.assertTrue(hasattr(LinearSVC(), "predict"))

    def test_model_has_decision_function(self):

        self.assertTrue(hasattr(LinearSVC(), "decision_function"))

    # ----- compute_metrics -----

    def test_compute_metrics_keys(self):

        model = LinearSVC().fit(X, Y)

        metrics = compute_metrics(model, X, Y)

        for key in ("y_pred", "accuracy", "f1", "auc"):
            self.assertIn(key, metrics)

    def test_compute_metrics_perfect_separation(self):

        model = LinearSVC().fit(X, Y)

        metrics = compute_metrics(model, X, Y)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(len(metrics["y_pred"]), len(Y))

    def test_compute_metrics_auc_from_decision_function(self):

        # LinearSVC exposes decision_function -> AUC should be computed.
        model = LinearSVC().fit(X, Y)

        metrics = compute_metrics(model, X, Y)

        self.assertIsNotNone(metrics["auc"])
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)

    def test_compute_metrics_auc_from_predict_proba(self):

        # LogisticRegression exposes predict_proba -> AUC via probabilities.
        model = LogisticRegression().fit(X, Y)

        metrics = compute_metrics(model, X, Y)

        self.assertIsNotNone(metrics["auc"])
        self.assertEqual(metrics["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
