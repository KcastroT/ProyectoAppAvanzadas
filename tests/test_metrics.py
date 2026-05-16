import unittest

from sklearn.svm import LinearSVC


class TestMetrics(unittest.TestCase):
    """
    Unit tests for validating model interface compatibility.
    """
    
    def test_model_has_predict(self):
        """
        Verify that the model implements a ``predict`` method.

        This ensures the classifier can generate predictions
        for input samples.
        """
        
        model = LinearSVC()

        self.assertTrue(
            hasattr(model, "predict")
        )

    def test_model_has_decision_function(self):
        """
        Verify that the model implements a ``decision_function`` method.

        This ensures the classifier supports decision scores,
        which can be used for metrics such as ROC AUC.
        """
        
        model = LinearSVC()

        self.assertTrue(
            hasattr(model, "decision_function")
        )


if __name__ == "__main__":
    unittest.main()
