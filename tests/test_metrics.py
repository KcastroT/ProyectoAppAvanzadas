import unittest

from sklearn.svm import LinearSVC


class TestMetrics(unittest.TestCase):

    def test_model_has_predict(self):

        model = LinearSVC()

        self.assertTrue(
            hasattr(model, "predict")
        )

    def test_model_has_decision_function(self):

        model = LinearSVC()

        self.assertTrue(
            hasattr(model, "decision_function")
        )


if __name__ == "__main__":
    unittest.main()
