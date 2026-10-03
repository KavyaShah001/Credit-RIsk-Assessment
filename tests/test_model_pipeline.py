"""End-to-end smoke test for fitting, serializing, and using the model bundle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.ingestion.csv_loader import RATIO_COLUMNS, TARGET_COLUMN
from credit_risk.models.predict import score_observation
from credit_risk.models.training import train_and_evaluate


class ModelPipelineTests(unittest.TestCase):
    def test_training_artifacts_and_single_observation_scoring(self) -> None:
        rng = np.random.default_rng(19)
        row_count = 240
        data = pd.DataFrame(rng.normal(size=(row_count, len(RATIO_COLUMNS))), columns=RATIO_COLUMNS)
        data["ratio_02"] = np.where(np.arange(row_count) % 7 == 0, 1.0, -0.3) + rng.normal(0, 0.1, row_count)
        data[TARGET_COLUMN] = (np.arange(row_count) % 6 == 0).astype("int8")
        data.loc[::13, "ratio_04"] = np.nan

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = train_and_evaluate(data, root / "models", root / "reports")
            bundle = joblib.load(root / "models" / "credit_risk_model.joblib")
            score = score_observation(bundle, data.loc[0, RATIO_COLUMNS].to_dict())

            self.assertIn(result["selected_model"], {"logistic_regression", "random_forest", "gradient_boosting"})
            self.assertTrue((root / "reports" / "model_card.json").is_file())
            self.assertTrue((root / "reports" / "roc_curve_test.png").is_file())
            holdout_scores = pd.read_csv(root / "reports" / "test_observation_scores.csv")
            self.assertEqual(len(holdout_scores), 48)
            self.assertFalse(holdout_scores["observation_id"].duplicated().any())
            self.assertEqual(score["outcome_proxy"], "bankruptcy")
            self.assertEqual(score["prediction_horizon_years"], 5)
            self.assertIn(score["internal_grade"], {"I1", "I2", "I3", "I4", "I5"})
            self.assertTrue(0.0 <= score["pd"] <= 1.0)
            self.assertTrue(np.isfinite(score["raw_pd"]))
            self.assertTrue(np.isfinite(score["calibrated_pd"]))


if __name__ == "__main__":
    unittest.main()
