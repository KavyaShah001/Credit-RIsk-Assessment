"""Behavioral guards against future-feature, temporal and entity leakage."""
import unittest
import numpy as np
import pandas as pd
from credit_risk.features.panel_features import build_panel_features, QuantileClipper, FEATURE_COLUMNS
from credit_risk.ingestion.panel_loader import validate_panel, FIELD_MAP
from credit_risk.evaluation.temporal_split import make_temporal_split, reserved_entities
from credit_risk.models.temporal_model import annual_grade
from credit_risk.stress_testing.statement_scenario import apply_margin_compression


def statements():
    rows = []
    for company in ['A', 'B']:
        for year in [2000, 2001, 2003]:
            row = dict.fromkeys(FIELD_MAP.values(), 10.)
            row.update(company_id=company, fiscal_year=year, total_assets=100., revenue=20.)
            rows.append(row)
    return pd.DataFrame(rows)


class TemporalIntegrityTests(unittest.TestCase):
    def test_cost_scenario_rebuilds_earnings_without_changing_history(self):
        history = statements().query("company_id == 'A'")
        unchanged, zero = apply_margin_compression(history, 2001, 0.)
        pd.testing.assert_frame_equal(history, unchanged)
        self.assertEqual(zero, 0.)
        stressed, cost = apply_margin_compression(history, 2001, 3.)
        self.assertAlmostEqual(cost, .6)
        self.assertAlmostEqual(stressed.loc[stressed.fiscal_year.eq(2001), 'ebitda'].iloc[0], 9.4)
        pd.testing.assert_frame_equal(history.loc[history.fiscal_year.ne(2001)], stressed.loc[stressed.fiscal_year.ne(2001)])

    def test_future_changes_cannot_change_past_features(self):
        base = statements()
        first = build_panel_features(base)
        base.loc[base.fiscal_year.eq(2003), list(FIELD_MAP.values())] = 999999.
        base['bankruptcy_next_year'] = 1
        changed = build_panel_features(base)
        pd.testing.assert_frame_equal(first.loc[base.fiscal_year.le(2001)], changed.loc[base.fiscal_year.le(2001)])
        self.assertNotIn('bankruptcy_next_year', FEATURE_COLUMNS)
        self.assertNotIn('company_id', FEATURE_COLUMNS)
        self.assertNotIn('fiscal_year', FEATURE_COLUMNS)

    def test_lags_do_not_cross_entities_or_year_gaps(self):
        frame = statements()
        out = build_panel_features(frame)
        self.assertTrue(out.loc[frame.fiscal_year.eq(2000), 'current_ratio_lag1'].isna().all())
        self.assertTrue(out.loc[frame.fiscal_year.eq(2003), 'current_ratio_lag1'].isna().all())
        self.assertTrue(out.loc[frame.fiscal_year.eq(2001), 'current_ratio_lag1'].eq(1).all())

    def test_nonpositive_denominators_are_missing_not_infinite(self):
        frame = statements()
        frame.loc[0, 'current_liabilities'] = 0.
        frame.loc[0, 'total_liabilities'] = 150.
        out = build_panel_features(frame)
        self.assertTrue(pd.isna(out.loc[0, 'current_ratio']))
        self.assertTrue(pd.isna(out.loc[0, 'return_on_equity']))
        self.assertEqual(out.loc[0, 'nonpositive_equity'], 1.)
        self.assertFalse(np.isinf(out.to_numpy()).any())

    def test_preprocessor_does_not_fit_on_transformed_future(self):
        clipper = QuantileClipper().fit(pd.DataFrame({'x': [0., 1., 2., 3.]}))
        original = clipper.upper_bounds_.copy()
        clipped = clipper.transform(pd.DataFrame({'x': [1e12]}))
        np.testing.assert_array_equal(clipper.upper_bounds_, original)
        self.assertLess(clipped[0, 0], 4)

    def test_entity_holdout_stable_under_row_order(self):
        ids = pd.Series([f'firm_{i}' for i in range(100)])
        mapping = dict(zip(ids, reserved_entities(ids, .2, 42)))
        reversed_ids = ids.iloc[::-1]
        self.assertEqual(mapping, dict(zip(reversed_ids, reserved_entities(reversed_ids, .2, 42))))

    def test_chronological_label_embargo_and_entity_exclusion(self):
        frame = pd.DataFrame([{'company_id': f'firm_{company}', 'fiscal_year': year,
                               'label_observed_year': year + 1, 'bankruptcy_next_year': company % 7 == 0}
                              for company in range(80) for year in range(2000, 2017)])
        config = dict(train_end=2005, calibration_start=2007, calibration_end=2008,
                      validation_start=2010, validation_end=2011, test_start=2013, test_end=2016,
                      unseen_entity_fraction=.2, random_seed=42)
        split, held, _ = make_temporal_split(frame, config)
        for name in ['train', 'calibration', 'validation']:
            self.assertFalse(held.loc[split[name]].any())
        for early, late in [('train', 'calibration'), ('calibration', 'validation'), ('validation', 'test')]:
            self.assertLess(frame.loc[split[early], 'label_observed_year'].max(),
                            frame.loc[split[late], 'fiscal_year'].min())
        with self.assertRaisesRegex(ValueError, 'outcome-maturity'):
            make_temporal_split(frame, {**config, 'calibration_start': 2006})

    def test_inference_rejects_duplicate_keys_and_invalid_years(self):
        frame = statements()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_panel(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        frame['fiscal_year'] = 2000.5
        with self.assertRaisesRegex(ValueError, 'integer fiscal years'):
            validate_panel(frame)
        with self.assertRaises(ValueError):
            annual_grade(float('nan'))

if __name__ == '__main__':
    unittest.main()
