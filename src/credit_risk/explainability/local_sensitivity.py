"""One-feature-at-a-time model sensitivity, not causal attribution."""

from __future__ import annotations

import pandas as pd


def local_ratio_sensitivity(
    model,
    observation: pd.DataFrame,
    training_medians: pd.Series,
    feature_names: list[str],
    baseline_pd: float | None = None,
) -> pd.DataFrame:
    """Measure score change when each observed ratio is set to its training median.

    The signed difference is scenario PD minus the observation's baseline PD.
    It is a local sensitivity check and does not establish causation or isolate
    correlated features.
    """
    if len(observation) != 1:
        raise ValueError("Provide exactly one observation for local sensitivity.")
    base_probability = (
        float(model.predict_proba(observation[feature_names])[:, 1][0])
        if baseline_pd is None
        else float(baseline_pd)
    )
    results = []
    for feature in feature_names:
        changed = observation[feature_names].copy()
        changed.loc[:, feature] = training_medians[feature]
        scenario_probability = float(model.predict_proba(changed)[:, 1][0])
        delta = scenario_probability - base_probability
        results.append(
            {
                "feature": feature,
                "observed_value": observation.iloc[0][feature],
                "training_median": training_medians[feature],
                "baseline_pd": base_probability,
                "pd_if_set_to_training_median": scenario_probability,
                "pd_change": delta,
                "interpretation": (
                    "Observed value is associated with higher model PD than its median-setting scenario"
                    if delta < 0
                    else "Observed value is associated with lower model PD than its median-setting scenario"
                    if delta > 0
                    else "No score change in this one-feature median-setting scenario"
                ),
            }
        )
    return pd.DataFrame(results).sort_values("pd_change", ascending=False, key=lambda values: values.abs())
