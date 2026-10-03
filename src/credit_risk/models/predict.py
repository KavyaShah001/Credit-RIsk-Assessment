"""Score an individual anonymous observation with the saved model bundle."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from credit_risk.models.risk_grades import grade_from_pd


def score_observation(bundle: Mapping[str, Any], values: Mapping[str, object]) -> dict[str, object]:
    """Return model-based raw/calibrated probabilities and the selected internal grade."""
    features = list(bundle["feature_columns"])
    row = pd.DataFrame([{feature: values.get(feature) for feature in features}], columns=features)
    raw_pd = float(bundle["raw_model"].predict_proba(row)[:, 1][0])
    calibrated_pd = float(bundle["calibrated_model"].predict_proba(row)[:, 1][0])
    selected_pd = calibrated_pd if bundle["use_calibrated_probability"] else raw_pd
    result = grade_from_pd(selected_pd)
    return {
        **result,
        "raw_pd": raw_pd,
        "calibrated_pd": calibrated_pd,
        "selected_probability_source": (
            "sigmoid calibrated"
            if bundle["use_calibrated_probability"]
            else "raw model probability; sigmoid calibration did not improve validation Brier score"
        ),
        "prediction_horizon_years": 5,
        "outcome_proxy": "bankruptcy",
        "selected_model": bundle["selected_model"],
    }
