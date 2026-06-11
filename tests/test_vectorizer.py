import unittest

from src.features.vectorizer import build_vectorizer, train_vectorizer


class TestVectorizer(unittest.TestCase):

    def test_build_vectorizer_config(self):

        vectorizer = build_vectorizer()

        # Lock the production config (character n-grams) so accidental
        # changes are caught.
        self.assertEqual(vectorizer.analyzer, "char_wb")
        self.assertEqual(vectorizer.ngram_range, (2, 5))
        self.assertEqual(vectorizer.max_features, 2000)
        self.assertEqual(vectorizer.min_df, 3)
        self.assertTrue(vectorizer.sublinear_tf)

    def test_vectorizer_fit_transform_shapes(self):

        vectorizer = build_vectorizer()

        # char_wb n-grams are shared across docs, so min_df=3 is easily met.
        X_train = [
            "hola mundo",
            "hola gente",
            "hola amigos",
            "machine learning",
        ]

        X_test = ["hola machine"]

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        self.assertEqual(X_train_tfidf.shape[0], 4)
        self.assertEqual(X_test_tfidf.shape[0], 1)
        # train and test share the same feature space.
        self.assertEqual(X_train_tfidf.shape[1], X_test_tfidf.shape[1])

    def test_tfidf_values_are_non_negative(self):

        vectorizer = build_vectorizer()

        X = ["hola mundo", "hola gente", "hola amigos"]

        tfidf = vectorizer.fit_transform(X)

        self.assertGreaterEqual(tfidf.min(), 0.0)

    def test_char_ngrams_are_produced(self):

        vectorizer = build_vectorizer()

        # "hol" (a 3-char gram) appears in >=3 docs -> survives min_df=3.
        X = ["hola a", "hola b", "hola c", "otra cosa"]

        vectorizer.fit(X)

        features = set(vectorizer.get_feature_names_out())

        self.assertIn("hol", features)

    def test_train_vectorizer_helper(self):

        vectorizer = build_vectorizer()

        X_train = ["hola mundo", "hola gente", "hola amigos"]
        X_test = ["hola otra"]

        X_train_tfidf, X_test_tfidf = train_vectorizer(
            vectorizer, X_train, X_test
        )

        self.assertEqual(X_train_tfidf.shape[0], 3)
        self.assertEqual(X_test_tfidf.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
