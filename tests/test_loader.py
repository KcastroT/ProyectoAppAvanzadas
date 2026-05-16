import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import (
    load_csv_dataset,
)


class TestLoader(unittest.TestCase):
    """
    Unit tests for dataset loading utilities.
    """
    
    def test_load_csv_dataset(self):
        """
        Verify that a CSV dataset is successfully loaded into a DataFrame.

        The test creates a temporary CSV file, loads it using
        ``load_csv_dataset()``, checks that the resulting DataFrame is not
        empty, and then removes the temporary file.
        """
        
        sample_path = Path("temp_test.csv")

        df = pd.DataFrame({
            "text": ["hello"]
        })

        df.to_csv(sample_path, index=False)

        loaded_df = load_csv_dataset(sample_path)

        self.assertFalse(loaded_df.empty)

        sample_path.unlink()


if __name__ == "__main__":
    unittest.main()
