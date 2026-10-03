"""Transparent demonstration-only internal grade mapping."""

from __future__ import annotations

GRADE_BANDS = (
    (0.01, "I1", "Lowest modeled risk"),
    (0.03, "I2", "Lower modeled risk"),
    (0.075, "I3", "Moderate modeled risk"),
    (0.15, "I4", "Elevated modeled risk"),
    (1.0000000001, "I5", "Highest modeled risk"),
)


def score_from_pd(pd_estimate: float) -> int:
    """Convert probability to an intuitive 0–100 score; higher means lower risk."""
    probability = float(min(max(pd_estimate, 0.0), 1.0))
    return int(round(100.0 * (1.0 - probability)))


def grade_from_pd(pd_estimate: float) -> dict[str, object]:
    """Map the selected five-year bankruptcy-proxy probability to an internal band."""
    probability = float(min(max(pd_estimate, 0.0), 1.0))
    for upper_bound, grade, label in GRADE_BANDS:
        if probability < upper_bound:
            return {
                "risk_score": score_from_pd(probability),
                "internal_grade": grade,
                "grade_label": label,
                "pd": probability,
            }
    raise AssertionError("Grade bands must cover the full probability range.")
