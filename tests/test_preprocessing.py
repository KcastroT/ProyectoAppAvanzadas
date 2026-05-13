import unittest

from src.data.preprocessing import (
    remove_urls,
    remove_mentions,
    normalize_hashtags,
    normalize_whitespace,
    clean_text,
)


class TestPreprocessing(unittest.TestCase):

    def test_remove_urls(self):

        text = "hello http://google.com"

        result = remove_urls(text)

        self.assertNotIn("http", result)

    def test_remove_mentions(self):

        text = "@kevin hello"

        result = remove_mentions(text)

        self.assertNotIn("@kevin", result)

    def test_normalize_hashtags(self):

        text = "#thinspo"

        result = normalize_hashtags(text)

        self.assertEqual(result, "thinspo")

    def test_normalize_whitespace(self):

        text = "hello     world"

        result = normalize_whitespace(text)

        self.assertEqual(result, "hello world")

    def test_clean_text_lowercase(self):

        text = "HELLO WORLD"

        result = clean_text(text)

        self.assertEqual(result, "hello world")

    def test_clean_text_none(self):

        result = clean_text(None)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
