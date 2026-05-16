import unittest

from src.data.preprocessing import (
    remove_urls,
    remove_mentions,
    normalize_hashtags,
    normalize_whitespace,
    clean_text,
)


class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for text preprocessing utility functions.
    """
    
    def test_remove_urls(self):
        """
        Verify that URLs are removed from text strings.
        """
        
        text = "hello http://google.com"

        result = remove_urls(text)

        self.assertNotIn("http", result)

    def test_remove_mentions(self):
        """
        Verify that user mentions are removed from text strings.
        """

        text = "@kevin hello"

        result = remove_mentions(text)

        self.assertNotIn("@kevin", result)

    def test_normalize_hashtags(self):
        """
        Verify that hashtags are normalized by removing the ``#`` symbol.
        """
        
        text = "#thinspo"

        result = normalize_hashtags(text)

        self.assertEqual(result, "thinspo")

    def test_normalize_whitespace(self):
        """
        Verify that multiple whitespace characters are reduced
        to single spaces.
        """
        
        text = "hello     world"

        result = normalize_whitespace(text)

        self.assertEqual(result, "hello world")

    def test_clean_text_lowercase(self):
        """
        Verify that text is converted to lowercase during cleaning.
        """
        
        text = "HELLO WORLD"

        result = clean_text(text)

        self.assertEqual(result, "hello world")

    def test_clean_text_none(self):
        """
        Verify that ``clean_text`` returns ``None`` when the input is ``None``.
        """
        
        result = clean_text(None)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
