"""Point-in-time financial analysis for the fields in the selected UCI source.

The source provides ratios, not statement line items. Direct metrics are therefore
reported as supplied; only two mathematically supportable ratios are derived.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricDefinition:
    code: str
    name: str
    category: str
    formula: str
    source_fields: tuple[str, ...]
    interpretation: str
    limitation: str = ""


@dataclass(frozen=True)
class MetricResult:
    code: str
    name: str
    category: str
    formula: str
    source_fields: tuple[str, ...]
    value: float | None
    status: str
    interpretation: str
    limitation: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "current_ratio", "Current ratio", "Liquidity",
        "Current assets / short-term liabilities", ("ratio_04",),
        "Compares current assets with short-term liabilities; composition and timing of working capital matter.",
        "Reported directly by the source; not recalculated from statement amounts.",
    ),
    MetricDefinition(
        "quick_ratio", "Quick ratio (source definition)", "Liquidity",
        "(Current assets - inventory) / short-term liabilities", ("ratio_46",),
        "Compares a narrower set of current assets with short-term liabilities; collectability and industry context matter.",
        "The source formula excludes inventory; it may not match every textbook quick-ratio convention.",
    ),
    MetricDefinition(
        "working_capital_to_assets", "Working capital / total assets", "Liquidity",
        "Working capital / total assets", ("ratio_03",),
        "Shows working capital relative to the asset base; it should be read with operating-cycle needs.",
        "Reported directly by the source.",
    ),
    MetricDefinition(
        "liabilities_to_assets", "Total liabilities / total assets", "Leverage",
        "Total liabilities / total assets", ("ratio_02",),
        "Shows the share of assets financed by liabilities; accounting structure and asset quality affect interpretation.",
        "This is liabilities-to-assets, not interest-bearing debt-to-assets.",
    ),
    MetricDefinition(
        "liabilities_to_equity", "Total liabilities / equity (derived)", "Leverage",
        "(Total liabilities / total assets) / (Equity / total assets)", ("ratio_02", "ratio_10"),
        "Relates total liabilities to equity; capital structure and the accounting basis should be considered.",
        "Derived from two source ratios; unavailable when equity / assets is missing or zero.",
    ),
    MetricDefinition(
        "long_term_liabilities_to_equity", "Long-term liabilities / equity", "Leverage",
        "Long-term liabilities / equity", ("ratio_59",),
        "Compares long-term liabilities with equity; maturity structure is not available in this dataset.",
        "Reported directly by the source.",
    ),
    MetricDefinition(
        "return_on_assets", "Return on assets (net-profit basis)", "Profitability",
        "Net profit / total assets", ("ratio_01",),
        "Relates net profit to assets employed; accounting policy and sector differences matter.",
        "Reported directly by the source.",
    ),
    MetricDefinition(
        "ebit_to_assets", "EBIT / total assets", "Profitability",
        "EBIT / total assets", ("ratio_07",),
        "Relates operating earnings before interest and tax to the asset base; it is not a margin.",
        "Reported directly by the source.",
    ),
    MetricDefinition(
        "operating_margin", "Operating profit / sales", "Profitability",
        "Profit on operating activities / sales", ("ratio_42",),
        "Shows operating profit relative to sales; business mix and accounting classification matter.",
        "Uses the source's operating-activities definition.",
    ),
    MetricDefinition(
        "net_profit_margin", "Net profit / sales", "Profitability",
        "Net profit / sales", ("ratio_23",),
        "Shows net profit relative to sales; one-period movements can reflect non-operating items.",
        "Reported directly by the source.",
    ),
    MetricDefinition(
        "return_on_equity", "Return on equity (derived)", "Profitability",
        "(Net profit / total assets) / (Equity / total assets)", ("ratio_01", "ratio_10"),
        "Relates net profit to equity; leverage can amplify the result and negative or near-zero equity complicates interpretation.",
        "Derived from source ratios; unavailable when equity / assets is missing or zero.",
    ),
    MetricDefinition(
        "operating_profit_to_financial_expenses", "Operating profit / financial expenses", "Coverage",
        "Profit on operating activities / financial expenses", ("ratio_27",),
        "A source coverage indicator comparing operating profit with financial expenses; earnings and expenses may be volatile.",
        "Not a fully standardized interest coverage ratio: the source specifies financial expenses, not interest expense alone.",
    ),
    MetricDefinition(
        "net_income_plus_depreciation_to_liabilities", "(Net profit + depreciation) / liabilities", "Cash-flow proxy",
        "(Net profit + depreciation) / total liabilities", ("ratio_26",),
        "A source earnings-plus-depreciation proxy relative to liabilities; it is not operating cash flow / debt.",
        "Cash flow statements are not supplied, so this must not be presented as an observed cash-flow ratio.",
    ),
    MetricDefinition(
        "gross_profit_plus_depreciation_to_liabilities", "(Gross profit + depreciation) / liabilities", "Cash-flow proxy",
        "(Gross profit + depreciation) / total liabilities", ("ratio_16",),
        "A source earnings-plus-depreciation proxy relative to liabilities; it is not free cash flow or debt service coverage.",
        "Cash flow statements and scheduled debt service are not supplied.",
    ),
)


UNAVAILABLE_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "cash_ratio", "Cash ratio", "Liquidity", "Cash / current liabilities", (),
        "Would compare cash available with current liabilities.",
        "Unavailable: the source does not provide the raw cash and current-liability amounts needed for this calculation.",
    ),
    MetricDefinition(
        "total_debt_to_ebitda", "Total debt / EBITDA", "Leverage", "Interest-bearing debt / EBITDA", (),
        "Would compare interest-bearing debt with an earnings measure.",
        "Unavailable: the source does not separately identify total interest-bearing debt and an aligned EBITDA amount.",
    ),
    MetricDefinition(
        "operating_cash_flow_to_debt", "Operating cash flow / debt", "Cash flow", "Operating cash flow / debt", (),
        "Would relate cash generated by operations to debt obligations.",
        "Unavailable: raw operating cash flow and debt amounts are not provided.",
    ),
    MetricDefinition(
        "free_cash_flow", "Free cash flow", "Cash flow", "Operating cash flow - capital expenditure", (),
        "Would approximate cash remaining after capital expenditure.",
        "Unavailable: operating cash flow and capital expenditure are not provided.",
    ),
    MetricDefinition(
        "debt_service_coverage", "Debt service coverage", "Coverage", "Cash available for debt service / scheduled debt service", (),
        "Would compare cash available with principal and interest due.",
        "Unavailable: cash available for debt service and debt-maturity schedules are not provided.",
    ),
)


def _number(value: object) -> float | None:
    """Convert finite numeric-like input to float; treat missing/nonfinite as absent."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _direct_metric(definition: MetricDefinition, values: Mapping[str, object]) -> MetricResult:
    field = definition.source_fields[0]
    value = _number(values.get(field))
    return MetricResult(
        code=definition.code,
        name=definition.name,
        category=definition.category,
        formula=definition.formula,
        source_fields=definition.source_fields,
        value=value,
        status="available" if value is not None else "missing in this observation",
        interpretation=definition.interpretation,
        limitation=definition.limitation,
    )


def _derived_division(definition: MetricDefinition, values: Mapping[str, object]) -> MetricResult:
    numerator = _number(values.get(definition.source_fields[0]))
    denominator = _number(values.get(definition.source_fields[1]))
    if numerator is None or denominator is None:
        result, status = None, "missing source ratio"
    elif denominator == 0:
        result, status = None, "undefined: zero denominator"
    else:
        result, status = numerator / denominator, "available (derived)"
    return MetricResult(
        code=definition.code,
        name=definition.name,
        category=definition.category,
        formula=definition.formula,
        source_fields=definition.source_fields,
        value=result,
        status=status,
        interpretation=definition.interpretation,
        limitation=definition.limitation,
    )


def analyze_financial_ratios(values: Mapping[str, object]) -> list[MetricResult]:
    """Return interpretable supported metrics, preserving missing and undefined states."""
    results: list[MetricResult] = []
    for definition in METRIC_DEFINITIONS:
        if len(definition.source_fields) == 1:
            results.append(_direct_metric(definition, values))
        else:
            results.append(_derived_division(definition, values))
    return results


def unavailable_financial_metrics() -> list[MetricResult]:
    """List useful requested metrics that this dataset cannot support."""
    return [
        MetricResult(
            code=definition.code,
            name=definition.name,
            category=definition.category,
            formula=definition.formula,
            source_fields=definition.source_fields,
            value=None,
            status="not available in selected dataset",
            interpretation=definition.interpretation,
            limitation=definition.limitation,
        )
        for definition in UNAVAILABLE_METRICS
    ]


def metrics_as_records(values: Mapping[str, object], include_unavailable: bool = True) -> list[dict[str, object]]:
    """Return dashboard/report-friendly dictionaries for one observation."""
    records = [result.to_dict() for result in analyze_financial_ratios(values)]
    if include_unavailable:
        records.extend(result.to_dict() for result in unavailable_financial_metrics())
    return records
