"""Contract tests for supported financial ratios and the internal grade bands."""

from __future__ import annotations

import unittest

from credit_risk.financial_analysis.ratio_analysis import analyze_financial_ratios
from credit_risk.models.risk_grades import grade_from_pd, score_from_pd


class FinancialAnalysisTests(unittest.TestCase):
    def test_supported_metrics_preserve_missing_and_undefined_values(self) -> None:
        results = {
            result.code: result
            for result in analyze_financial_ratios(
                {
                    "ratio_01": 0.08,
                    "ratio_02": 0.60,
                    "ratio_07": 0.10,
                    "ratio_10": 0.20,
                    "ratio_42": -0.02,
                    "ratio_23": None,
                    "ratio_27": float("inf"),
                }
            )
        }

        self.assertAlmostEqual(results["liabilities_to_equity"].value, 3.0)
        self.assertAlmostEqual(results["return_on_equity"].value, 0.4)
        self.assertIsNone(results["net_profit_margin"].value)
        self.assertEqual(results["operating_profit_to_financial_expenses"].status, "missing in this observation")

    def test_derived_metric_marks_zero_denominator_undefined(self) -> None:
        results = {result.code: result for result in analyze_financial_ratios({"ratio_01": 0.1, "ratio_10": 0.0})}

        self.assertIsNone(results["return_on_equity"].value)
        self.assertEqual(results["return_on_equity"].status, "undefined: zero denominator")


class RiskGradeTests(unittest.TestCase):
    def test_grade_boundaries_and_score_direction(self) -> None:
        expected = {
            0.0: ("I1", 100),
            0.01: ("I2", 99),
            0.03: ("I3", 97),
            0.075: ("I4", 92),
            0.15: ("I5", 85),
            1.0: ("I5", 0),
        }
        for probability, (grade, score) in expected.items():
            with self.subTest(probability=probability):
                result = grade_from_pd(probability)
                self.assertEqual(result["internal_grade"], grade)
                self.assertEqual(result["risk_score"], score)
                self.assertEqual(result["pd"], probability)

        self.assertEqual(score_from_pd(-1.0), 100)
        self.assertEqual(score_from_pd(2.0), 0)


if __name__ == "__main__":
    unittest.main()
