import unittest

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

from src.data.preprocessing import (
    SLANG_MAP,
    add_clean_text_column,
    apply_stemming,
    clean_text,
    clean_text_light,
    expand_slang,
    fix_dataframe_encoding,
    fix_encoding,
    normalize_hashtags,
    normalize_whitespace,
    remove_mentions,
    remove_urls,
)


class TestPreprocessing(unittest.TestCase):

    # ----- individual helpers -----

    def test_remove_urls(self):

        self.assertNotIn("http", remove_urls("hello http://google.com"))

    def test_remove_urls_www(self):

        self.assertNotIn("www", remove_urls("see www.example.com now"))

    def test_remove_mentions(self):

        self.assertNotIn("@kevin", remove_mentions("@kevin hello"))

    def test_normalize_hashtags(self):

        self.assertEqual(normalize_hashtags("#thinspo"), "thinspo")

    def test_normalize_hashtags_multiple(self):

        self.assertEqual(normalize_hashtags("#a #b"), "a b")

    def test_normalize_whitespace(self):

        self.assertEqual(normalize_whitespace("hello     world"), "hello world")

    def test_normalize_whitespace_strips_edges(self):

        self.assertEqual(normalize_whitespace("  hello  "), "hello")

    # ----- stemming -----

    def test_apply_stemming_matches_snowball(self):

        stemmer = SnowballStemmer("spanish")

        self.assertEqual(
            apply_stemming("comiendo"),
            stemmer.stem("comiendo"),
        )

    def test_apply_stemming_preserves_word_count(self):

        result = apply_stemming("gatos perros casas")

        self.assertEqual(len(result.split()), 3)

    # ----- slang expansion -----

    def test_expand_slang_known_term(self):

        self.assertEqual(expand_slang("thinspo", SLANG_MAP), "inspiracion delgada")

    def test_expand_slang_case_insensitive(self):

        self.assertEqual(expand_slang("ANA", SLANG_MAP), "anorexia")

    def test_expand_slang_keeps_unknown(self):

        self.assertEqual(expand_slang("bicicleta", SLANG_MAP), "bicicleta")

    # ----- clean_text (aggressive: lowercase + stemming) -----

    def test_clean_text_lowercases(self):

        # clean_text stems, so don't assert an exact string; just verify it
        # is fully lowercased and case-insensitive.
        result = clean_text("HELLO WORLD")

        self.assertTrue(result.islower())
        self.assertEqual(result, clean_text("hello world"))

    def test_clean_text_removes_noise(self):

        result = clean_text("Hola @user mira http://x.com #salud")

        self.assertNotIn("http", result)
        self.assertNotIn("@user", result)
        self.assertNotIn("#", result)

    def test_clean_text_none(self):

        self.assertIsNone(clean_text(None))

    # ----- clean_text_light (keeps case, no stemming) -----

    def test_clean_text_light_keeps_case(self):

        result = clean_text_light("Hola @user http://x.com #Salud")

        self.assertEqual(result, "Hola Salud")

    def test_clean_text_light_none(self):

        self.assertIsNone(clean_text_light(None))

    # ----- encoding -----

    def test_fix_encoding_repairs_mojibake(self):

        self.assertEqual(fix_encoding("Ã©"), "é")

    def test_fix_encoding_passthrough_non_string(self):

        self.assertEqual(fix_encoding(123), 123)

    def test_fix_dataframe_encoding(self):

        df = pd.DataFrame({"text": ["Ã©xito"], "n": [1]})

        fixed = fix_dataframe_encoding(df)

        self.assertEqual(fixed.loc[0, "text"], "éxito")
        # numeric column untouched
        self.assertEqual(fixed.loc[0, "n"], 1)

    # ----- dataframe column helper -----

    def test_add_clean_text_column_adds_column(self):

        df = pd.DataFrame({"raw": ["HELLO http://x.com"]})

        out = add_clean_text_column(df, "raw", "clean")

        self.assertIn("clean", out.columns)
        self.assertNotIn("http", out.loc[0, "clean"])

    def test_add_clean_text_column_does_not_mutate_original(self):

        df = pd.DataFrame({"raw": ["hola"]})

        add_clean_text_column(df, "raw", "clean")

        self.assertNotIn("clean", df.columns)


if __name__ == "__main__":
    unittest.main()
