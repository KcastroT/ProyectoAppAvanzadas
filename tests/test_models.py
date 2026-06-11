import os
import sys
import unittest

# The model builders do ``from config import ...`` (assuming src/ is on the
# path, as when main.py runs from src/), so make src importable here too.
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from models.logistic_regression_model import build_model as build_lr
from models.random_forest_model import build_model as build_rf
from models.svm_model import build_model as build_svm

# Tiny separable dataset to confirm the estimators actually fit/predict.
X = [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]]
Y = ["control", "control", "control", "anorexia", "anorexia", "anorexia"]


class TestModelBuilders(unittest.TestCase):

    def test_build_svm_type_and_config(self):

        model = build_svm()

        self.assertIsInstance(model, LinearSVC)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)

    def test_build_rf_type_and_config(self):

        model = build_rf()

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)
        self.assertGreaterEqual(model.n_estimators, 1)

    def test_build_lr_type_and_config(self):

        model = build_lr()

        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)

    def test_lr_exposes_predict_proba(self):

        # LogisticRegression is the grid's probabilistic cell (AUC via proba).
        self.assertTrue(hasattr(build_lr(), "predict_proba"))

    def test_builders_fit_and_predict(self):

        for build in (build_svm, build_rf, build_lr):
            model = build().fit(X, Y)

            preds = model.predict(X)

            self.assertEqual(len(preds), len(Y))
            self.assertTrue(set(preds).issubset({"control", "anorexia"}))

    def test_builders_return_fresh_instances(self):

        self.assertIsNot(build_svm(), build_svm())


if __name__ == "__main__":
    unittest.main()
