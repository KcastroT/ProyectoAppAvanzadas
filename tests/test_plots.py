import tempfile
import unittest
from pathlib import Path

from src.evaluation.plots import (
    save_cv_vs_test,
    save_model_comparison,
    save_overfit_synthesis,
)

LABELS = ["Modelo A", "Modelo B", "Modelo C"]


class TestPlots(unittest.TestCase):

    def setUp(self):

        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):

        self._dir.cleanup()

    def _assert_png(self, path):

        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_save_model_comparison_writes_png(self):

        out = self.tmp / "model_comparison.png"

        save_model_comparison(
            LABELS, [0.80, 0.85, 0.90], [0.01, 0.02, 0.015], out, "Comparación"
        )

        self._assert_png(out)

    def test_save_cv_vs_test_writes_png(self):

        out = self.tmp / "cv_vs_test.png"

        save_cv_vs_test(
            LABELS, [0.80, 0.85, 0.90], [0.01, 0.02, 0.015],
            [0.79, 0.86, 0.88], out, "CV vs Test",
        )

        self._assert_png(out)

    def test_save_overfit_synthesis_writes_png(self):

        out = self.tmp / "overfit.png"

        save_overfit_synthesis(
            LABELS, [1.0, 0.95, 1.0], [0.85, 0.86, 0.87], out, "Sobreajuste"
        )

        self._assert_png(out)


if __name__ == "__main__":
    unittest.main()
