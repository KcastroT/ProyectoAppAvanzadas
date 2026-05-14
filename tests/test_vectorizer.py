import unittest

from src.features.vectorizer import (
    build_vectorizer,
)


class TestVectorizer(unittest.TestCase):

    def test_build_vectorizer(self):

        vectorizer = build_vectorizer()

        self.assertEqual(
            vectorizer.max_features,
            10000,
        )

    def test_vectorizer_fit_transform(self):

        vectorizer = build_vectorizer()

        X_train = [
            "hello world",
            "machine learning",
        ]

        X_test = [
            "hello machine"
        ]

        X_train_tfidf = (
            vectorizer.fit_transform(X_train)
        )

        X_test_tfidf = (
            vectorizer.transform(X_test)
        )

        self.assertEqual(
            X_train_tfidf.shape[0],
            2,
        )

        self.assertEqual(
            X_test_tfidf.shape[0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
