import unittest

import numpy as np

from src.models.llm import OllamaClassifier, build_model

# These tests never hit the network: they exercise fit (which only records
# labels and samples few-shot examples) and the JSON label parsing / fallback.

X = np.array(
    ["no como nada", "odio mi cuerpo", "comi sano hoy", "fui al cine"],
    dtype=object,
)
Y = np.array(["anorexia", "anorexia", "control", "control"])


def fitted(**kwargs):
    clf = OllamaClassifier(**kwargs)
    clf.fit(X, Y)
    return clf


class TestOllamaClassifier(unittest.TestCase):

    # ----- fit / label bookkeeping -----

    def test_fit_sets_sorted_classes(self):

        clf = fitted()

        self.assertEqual(list(clf.classes_), ["anorexia", "control"])

    def test_fit_returns_self(self):

        clf = OllamaClassifier()

        self.assertIs(clf.fit(X, Y), clf)

    def test_fallback_label_is_first_class(self):

        clf = fitted()

        self.assertEqual(clf._fallback_label, "anorexia")

    # ----- few-shot sampling -----

    def test_few_shot_count(self):

        clf = fitted(n_few_shot=2)

        # 2 per class x 2 classes = 4 examples.
        self.assertEqual(len(clf._few_shot), 4)

    def test_zero_shot_has_no_examples(self):

        clf = fitted(n_few_shot=0)

        self.assertEqual(clf._few_shot, [])

    def test_few_shot_capped_by_available(self):

        clf = fitted(n_few_shot=10)

        # Only 2 examples per class exist, so 4 total.
        self.assertEqual(len(clf._few_shot), 4)

    # ----- label parsing -----

    def test_parse_label_exact(self):

        clf = fitted()

        self.assertEqual(clf._parse_label('{"label": "control"}'), "control")

    def test_parse_label_case_insensitive(self):

        clf = fitted()

        self.assertEqual(clf._parse_label('{"label": "CONTROL"}'), "control")

    def test_parse_label_substring(self):

        clf = fitted()

        self.assertEqual(
            clf._parse_label('{"label": "anorexia (TCA)"}'), "anorexia"
        )

    def test_parse_label_unknown_falls_back(self):

        clf = fitted()

        self.assertEqual(clf._parse_label('{"label": "otro"}'), "anorexia")

    def test_parse_label_missing_key_falls_back(self):

        clf = fitted()

        self.assertEqual(clf._parse_label("{}"), "anorexia")

    # ----- probability parsing (for AUC) -----

    def test_parse_prob_reads_value(self):

        clf = fitted()

        content = '{"label": "anorexia", "prob_anorexia": 0.8}'

        self.assertEqual(clf._parse_prob(content, "anorexia"), 0.8)

    def test_parse_prob_missing_uses_label(self):

        clf = fitted()

        # No prob field -> coarse value implied by the (positive) label.
        self.assertEqual(clf._parse_prob('{"label": "anorexia"}', "anorexia"), 0.85)

    def test_parse_prob_out_of_range_falls_back(self):

        clf = fitted()

        self.assertEqual(
            clf._parse_prob('{"prob_anorexia": 5}', "control"), 0.15
        )

    def test_prob_from_label(self):

        clf = fitted()

        self.assertEqual(clf._prob_from_label("anorexia"), 0.85)
        self.assertEqual(clf._prob_from_label("control"), 0.15)

    # ----- truncation -----

    def test_truncate_collapses_whitespace(self):

        clf = OllamaClassifier()

        self.assertEqual(clf._truncate("hola    mundo\n hey"), "hola mundo hey")

    def test_truncate_respects_max_chars(self):

        clf = OllamaClassifier(max_chars=5)

        self.assertEqual(len(clf._truncate("abcdefghij")), 5)

    # ----- build_model factory -----

    def test_build_model_returns_classifier(self):

        clf = build_model(model_name="qwen2.5:3b", n_few_shot=3)

        self.assertIsInstance(clf, OllamaClassifier)
        self.assertEqual(clf.model_name, "qwen2.5:3b")
        self.assertEqual(clf.n_few_shot, 3)


if __name__ == "__main__":
    unittest.main()
