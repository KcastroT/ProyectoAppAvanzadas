import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import (
    load_csv_dataset,
)


class TestLoader(unittest.TestCase):

    def test_load_csv_dataset(self):

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
