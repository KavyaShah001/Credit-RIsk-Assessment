"""Build a standalone, downloadable HTML observation credit-risk summary."""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence


def _cell(value: object) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_observation_report(
    observation_id: str,
    risk: Mapping[str, object],
    financial_metrics: Sequence[Mapping[str, object]],
    risk_increasing: Sequence[Mapping[str, object]],
    risk_reducing: Sequence[Mapping[str, object]],
    scenario: Mapping[str, object] | None = None,
) -> str:
    """Return self-contained HTML with escaped input data and project caveats."""

    def metric_rows() -> str:
        rows = []
        for metric in financial_metrics:
            value = metric.get("value")
            rendered = f"{float(value):.5g}" if value is not None else "Not available"
            rows.append(
                "<tr>"
                f"<td>{escape(str(metric.get('name', '')))}</td>"
                f"<td>{escape(rendered)}</td>"
                f"<td>{escape(str(metric.get('formula', '')))}</td>"
                f"<td>{escape(str(metric.get('status', '')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def driver_rows(drivers: Sequence[Mapping[str, object]]) -> str:
        return "\n".join(
            "<li>"
            f"{escape(str(driver.get('source_definition', driver.get('feature', ''))))} "
            f"(one-feature median sensitivity: {float(driver.get('risk_effect', 0.0)) * 100:+.3f} pp)"
            "</li>"
            for driver in drivers
        ) or "<li>None identified in the displayed sensitivity ranking.</li>"

    scenario_section = "<p>No stress scenario was run.</p>"
    if scenario:
        scenario_section = (
            "<p>Single-ratio sensitivity, not a full financial statement stress test: "
            f"{escape(str(scenario['label']))}. Base PD {float(scenario['base_pd']):.2%}; "
            f"scenario PD {float(scenario['scenario_pd']):.2%}; "
            f"change {float(scenario['change'])*100:+.2f} percentage points.</p>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Risk Summary — {escape(observation_id)}</title>
<style>body{{font:15px/1.55 Arial,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;color:#17202a}}h1,h2{{color:#16324f}}table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left}}th{{background:#eef3f8}}.notice{{background:#fff7e6;padding:12px;border-left:4px solid #dc9b21}}</style>
</head><body>
<h1>Credit Risk Assessment</h1><p><strong>Anonymous observation:</strong> {escape(observation_id)}</p>
<h2>Credit-risk assessment</h2><ul>
<li>Five-year bankruptcy proxy PD: <strong>{float(risk['pd']):.2%}</strong></li>
<li>Project Internal Credit Risk Rating: <strong>{escape(str(risk['internal_grade']))}</strong> — {escape(str(risk['grade_label']))}</li>
<li>Risk score: {escape(str(risk['risk_score']))} / 100 (higher indicates lower modeled risk)</li>
<li>Model: {escape(str(risk['selected_model']))}; probability: {escape(str(risk['selected_probability_source']))}</li>
</ul>
<h2>Financial health indicators</h2><table><thead><tr><th>Metric</th><th>Value</th><th>Formula</th><th>Status</th></tr></thead><tbody>{metric_rows()}</tbody></table>
<h2>Risk drivers</h2><p>One-feature-at-a-time comparison with training-set medians; these are model sensitivities, not causal contributions.</p>
<h3>Risk-increasing associations</h3><ul>{driver_rows(risk_increasing)}</ul>
<h3>Risk-reducing associations</h3><ul>{driver_rows(risk_reducing)}</ul>
<h2>Scenario result</h2>{scenario_section}
<h2>Limitations</h2><div class="notice"><ul>
<li>This dataset contains anonymous observations, not named counterparties, sectors, or exposures.</li>
<li>The label is bankruptcy within five years, used as a proxy; the result is not a contractual default PD.</li>
<li>Internal grades are project assumptions and do not represent an external rating scale.</li>
<li>The holdout is a stratified random split, not a temporal backtest. The source has no dates or stable entity identifiers.</li>
<li>Financial ratios are indicators whose meaning depends on industry, company profile, accounting policy, and context.</li>
</ul></div></body></html>"""
