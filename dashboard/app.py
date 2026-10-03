"""Streamlit interface for the credit-risk analytics MVP."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from credit_risk.explainability.local_sensitivity import local_ratio_sensitivity
from credit_risk.financial_analysis.ratio_analysis import analyze_financial_ratios
from credit_risk.financial_analysis.ratio_dictionary import RATIO_DICTIONARY
from credit_risk.ingestion.csv_loader import RATIO_COLUMNS
from credit_risk.models.predict import score_observation
from credit_risk.reporting.html_report import render_observation_report

DATABASE_PATH = ROOT / "data" / "processed" / "credit_risk.sqlite3"
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
REPORT_PATH = ROOT / "reports"
PREDICTION_PATH = REPORT_PATH / "test_observation_scores.csv"
RATIO_METADATA = {item["code"]: item for item in RATIO_DICTIONARY}

st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading held-out observations from SQLite…")
def load_dashboard_data() -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError("SQLite data is missing. Run scripts/load_sqlite.py first.")
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError("Held-out scores are missing. Run scripts/train_models.py first.")

    with sqlite3.connect(DATABASE_PATH) as connection:
        wide = pd.read_sql_query(
            """SELECT c.company_id AS observation_id,
                      c.source_row_number,
                      f.ratio_code,
                      f.ratio_value
               FROM companies AS c
               JOIN financial_observations AS o ON o.company_id = c.company_id
               JOIN financial_ratios AS f ON f.observation_id = o.observation_id
               WHERE c.source_dataset = 'uci_polish_1year'
               ORDER BY c.source_row_number, f.ratio_code""",
            connection,
        )

    ratios = wide.pivot(index=["observation_id", "source_row_number"], columns="ratio_code", values="ratio_value")
    ratios = ratios.reindex(columns=RATIO_COLUMNS).reset_index()
    scores = pd.read_csv(PREDICTION_PATH)
    dashboard = scores.merge(ratios, on="observation_id", how="inner", validate="one_to_one")
    if dashboard.empty:
        raise ValueError("No held-out model scores matched the SQLite observations.")
    return dashboard


def risk_for_row(bundle, row: pd.Series) -> dict[str, object]:
    inputs = {feature: row.get(feature) for feature in RATIO_COLUMNS}
    return score_observation(bundle, inputs)


def sensitivity_for_row(bundle, row: pd.Series, risk: dict[str, object]) -> pd.DataFrame:
    selected_model = bundle["calibrated_model"] if bundle["use_calibrated_probability"] else bundle["raw_model"]
    feature_row = pd.DataFrame([{column: row.get(column) for column in RATIO_COLUMNS}])
    result = local_ratio_sensitivity(
        selected_model,
        feature_row,
        bundle["training_medians"],
        RATIO_COLUMNS,
        baseline_pd=float(risk["pd"]),
    )
    result["source_definition"] = result["feature"].map(
        lambda code: RATIO_METADATA.get(code, {}).get("description", code)
    )
    result["risk_effect"] = -result["pd_change"]
    return result


def dataframe_download(label: str, report_html: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=report_html.encode("utf-8"),
        file_name=filename,
        mime="text/html",
        width="stretch",
    )


def show_portfolio(data: pd.DataFrame) -> None:
    st.header("Portfolio overview")
    st.caption(
        "This demonstration portfolio consists of anonymous observations from the model's held-out test split. "
        "The dataset has no named counterparties, sectors, or exposure amounts."
    )
    grade_options = ["I1", "I2", "I3", "I4", "I5"]
    selected_grades = st.multiselect("Include internal grades", grade_options, default=grade_options)
    view = data[data["internal_grade"].isin(selected_grades)]
    if view.empty:
        st.info("Select at least one grade to display the portfolio view.")
        return

    first, second, third, fourth = st.columns(4)
    first.metric("Observations", f"{len(view):,}")
    second.metric("Average modeled PD", f"{view['selected_pd'].mean():.2%}")
    third.metric("Median modeled PD", f"{view['selected_pd'].median():.2%}")
    fourth.metric("Grade I4–I5 share", f"{view['internal_grade'].isin(['I4', 'I5']).mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Project internal grade distribution")
        grade_counts = view["internal_grade"].value_counts().reindex(grade_options, fill_value=0)
        st.bar_chart(grade_counts.rename("Observations"))
    with right:
        st.subheader("Five-year bankruptcy-proxy PD distribution")
        histogram = pd.cut(
            view["selected_pd"],
            bins=[-0.000001, 0.01, 0.03, 0.075, 0.15, 1.000001],
            labels=["<1%", "1–3%", "3–7.5%", "7.5–15%", "15%+"],
            include_lowest=True,
        ).value_counts(sort=False)
        st.bar_chart(histogram.rename("Observations"))

    st.subheader("Highest modeled risk in the selected view")
    high_risk = view.nlargest(10, "selected_pd")[["observation_id", "selected_pd", "internal_grade", "risk_score"]].copy()
    high_risk["selected_pd"] = high_risk["selected_pd"].map(lambda value: f"{value:.2%}")
    st.dataframe(high_risk, hide_index=True, width="stretch")
    st.info(
        "Sector concentration and expected loss (PD × LGD × EAD) are not calculated: sector and exposure "
        "are absent, and LGD/EAD would require assumptions. These counts and PD summaries are not exposure-weighted."
    )


def show_company_analysis(data: pd.DataFrame, bundle) -> None:
    st.header("Observation credit analysis")
    st.caption("The source is anonymized. IDs below are technical row references, not company names.")
    observation_id = st.selectbox("Select an anonymous observation", data["observation_id"].tolist())
    row = data.loc[data["observation_id"].eq(observation_id)].iloc[0]
    risk = risk_for_row(bundle, row)
    metrics = [metric.to_dict() for metric in analyze_financial_ratios(row.to_dict())]

    pd_col, grade_col, score_col, model_col = st.columns(4)
    pd_col.metric("Estimated five-year bankruptcy-proxy PD", f"{risk['pd']:.2%}")
    grade_col.metric("Project Internal Credit Risk Rating", str(risk["internal_grade"]))
    score_col.metric("Risk score", f"{risk['risk_score']} / 100")
    model_col.metric("Selected model", str(risk["selected_model"]).replace("_", " ").title())
    st.caption(f"Probability source: {risk['selected_probability_source']}.")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Supported financial health indicators")
        metric_frame = pd.DataFrame(metrics)
        metric_frame["value"] = metric_frame["value"].map(lambda value: f"{value:.5g}" if value is not None else "Not available")
        st.dataframe(
            metric_frame[["name", "category", "value", "formula", "status"]],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Risk-driver sensitivity")
        st.caption("Change in model PD when each ratio is replaced with its training-set median. Associations, not causes.")
        drivers = sensitivity_for_row(bundle, row, risk)
        increases = drivers[drivers["risk_effect"] > 0].nlargest(5, "risk_effect")
        decreases = drivers[drivers["risk_effect"] < 0].nsmallest(5, "risk_effect")
        st.markdown("**Associated with higher PD than the median comparison**")
        if increases.empty:
            st.write("No higher-risk sensitivities in the top-five comparison.")
        else:
            table = increases[["feature", "source_definition", "risk_effect"]].copy()
            table["risk_effect"] = table["risk_effect"].map(lambda value: f"{value * 100:+.3f} pp")
            st.dataframe(table, hide_index=True, width="stretch")
        st.markdown("**Associated with lower PD than the median comparison**")
        if decreases.empty:
            st.write("No lower-risk sensitivities in the top-five comparison.")
        else:
            table = decreases[["feature", "source_definition", "risk_effect"]].copy()
            table["risk_effect"] = table["risk_effect"].map(lambda value: f"{value * 100:+.3f} pp")
            st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Source ratio fields"):
        fields = pd.DataFrame(
            [
                {"Ratio": code, "Value": row.get(code), "Source definition": RATIO_METADATA.get(code, {}).get("description", code)}
                for code in RATIO_COLUMNS
            ]
        )
        st.dataframe(fields, hide_index=True, width="stretch")
    st.warning(
        "The source contains ratio fields, not full financial statements. Missing metrics remain unavailable; "
        "risk interpretation depends on business and industry context."
    )

    report_html = render_observation_report(
        observation_id=observation_id,
        risk=risk,
        financial_metrics=metrics,
        risk_increasing=increases.to_dict(orient="records"),
        risk_reducing=decreases.to_dict(orient="records"),
    )
    dataframe_download("Download HTML credit-risk report", report_html, f"credit-risk-{observation_id}.html")


STRESS_SCENARIOS = {
    "Higher liabilities / assets": ("ratio_02", "higher"),
    "Lower current ratio": ("ratio_04", "lower"),
    "Lower quick ratio": ("ratio_46", "lower"),
    "Lower net profit / assets": ("ratio_01", "lower"),
    "Lower operating margin": ("ratio_42", "lower"),
    "Lower operating profit / financial expenses": ("ratio_27", "lower"),
}


def show_stress_testing(data: pd.DataFrame, bundle) -> None:
    st.header("Ratio-level sensitivity scenario")
    st.caption(
        "This is a single-ratio sensitivity, not a full revenue/debt financial-statement stress test. "
        "The dataset has no raw statement line items. Scenario direction is an explicit analyst assumption, "
        "and ratio interpretation is context-dependent."
    )
    observation_id = st.selectbox("Select an anonymous observation", data["observation_id"].tolist(), key="stress_observation")
    scenario_label = st.selectbox("Adverse ratio scenario", list(STRESS_SCENARIOS))
    severity = st.select_slider("Adverse percentile severity", options=[75, 90, 95, 99], value=90)

    row = data.loc[data["observation_id"].eq(observation_id)].iloc[0]
    ratio_code, direction = STRESS_SCENARIOS[scenario_label]
    ratio_values = pd.to_numeric(data[ratio_code], errors="coerce").dropna()
    adverse_quantile = severity / 100 if direction == "higher" else 1 - severity / 100
    quantile_value = float(ratio_values.quantile(adverse_quantile))
    observed_value = row.get(ratio_code)
    if pd.isna(observed_value):
        stressed_value = quantile_value
        original_state = "missing; model's imputation is used in the base case"
    elif direction == "higher":
        stressed_value = max(float(observed_value), quantile_value)
        original_state = f"observed {float(observed_value):.5g}"
    else:
        stressed_value = min(float(observed_value), quantile_value)
        original_state = f"observed {float(observed_value):.5g}"

    base_risk = risk_for_row(bundle, row)
    shocked_row = row.copy()
    shocked_row[ratio_code] = stressed_value
    scenario_risk = risk_for_row(bundle, shocked_row)
    change = float(scenario_risk["pd"]) - float(base_risk["pd"])
    relative_change = change / float(base_risk["pd"]) if float(base_risk["pd"]) > 0 else np.nan
    metadata = RATIO_METADATA[ratio_code]

    st.markdown(f"**Scenario:** {scenario_label} · {metadata['description']}")
    st.write(f"Input ratio: {original_state}; scenario ratio: **{stressed_value:.5g}** (held-out sample adverse percentile {severity}%).")
    a, b, c, d = st.columns(4)
    a.metric("Base PD", f"{base_risk['pd']:.2%}")
    b.metric("Scenario PD", f"{scenario_risk['pd']:.2%}")
    c.metric("Absolute change", f"{change * 100:+.2f} pp")
    d.metric("Relative change", f"{relative_change:+.1%}" if np.isfinite(relative_change) else "Not defined")
    st.write(
        f"Grade: **{base_risk['internal_grade']} → {scenario_risk['internal_grade']}**. "
        "The actual stressed PD is generated by the selected fitted model. A ratio shock is not a causal forecast."
    )

    metrics = [metric.to_dict() for metric in analyze_financial_ratios(shocked_row.to_dict())]
    sensitivity = sensitivity_for_row(bundle, row, base_risk)
    risk_increasing = sensitivity[sensitivity["risk_effect"] > 0].nlargest(5, "risk_effect").to_dict(orient="records")
    risk_reducing = sensitivity[sensitivity["risk_effect"] < 0].nsmallest(5, "risk_effect").to_dict(orient="records")
    report_html = render_observation_report(
        observation_id=observation_id,
        risk=base_risk,
        financial_metrics=metrics,
        risk_increasing=risk_increasing,
        risk_reducing=risk_reducing,
        scenario={
            "label": f"{scenario_label}; {metadata['description']} set to {stressed_value:.5g}",
            "base_pd": float(base_risk["pd"]),
            "scenario_pd": float(scenario_risk["pd"]),
            "change": change,
        },
    )
    dataframe_download("Download HTML report with scenario", report_html, f"credit-risk-{observation_id}-scenario.html")


def show_model_performance() -> None:
    st.header("Model performance and validation")
    st.caption("Model selection and operating threshold use validation data; test plots and metrics are held out.")
    required_files = [
        "model_comparison_validation.csv",
        "selected_model_raw_vs_calibrated_test.csv",
        "selected_model_test_at_validation_threshold.csv",
        "permutation_importance_validation.csv",
    ]
    missing = [name for name in required_files if not (REPORT_PATH / name).exists()]
    if missing:
        st.error("Model artifacts are missing. Run scripts/train_models.py before starting the dashboard.")
        return

    comparison = pd.read_csv(REPORT_PATH / "model_comparison_validation.csv")
    st.subheader("Validation candidate comparison (0.50 threshold)")
    st.dataframe(comparison, hide_index=True, width="stretch")
    raw_calibrated = pd.read_csv(REPORT_PATH / "selected_model_raw_vs_calibrated_test.csv")
    threshold_results = pd.read_csv(REPORT_PATH / "selected_model_test_at_validation_threshold.csv")
    st.subheader("Raw and calibrated probability comparison on test")
    st.dataframe(raw_calibrated, hide_index=True, width="stretch")
    st.caption(f"At the validation-selected operating threshold ({threshold_results.iloc[0]['threshold']:.4f}):")
    st.dataframe(threshold_results, hide_index=True, width="stretch")

    plots = [
        ("roc_curve_test.png", "ROC curve"),
        ("precision_recall_curve_test.png", "Precision–recall curve"),
        ("calibration_curve_test.png", "Calibration curve"),
        ("confusion_matrix_test.png", "Confusion matrix"),
    ]
    columns = st.columns(2)
    for index, (filename, label) in enumerate(plots):
        path = REPORT_PATH / filename
        if path.exists():
            with columns[index % 2]:
                st.image(str(path), caption=label, width="stretch")

    importance = pd.read_csv(REPORT_PATH / "permutation_importance_validation.csv").head(15).set_index("feature")
    st.subheader("Top global permutation importance (validation PR-AUC decrease)")
    st.caption("Higher means the validation score fell more when that feature was shuffled. This measures model reliance, not causation.")
    st.bar_chart(importance["importance_mean_pr_auc_decrease"])
    with st.expander("Definitions and limitations"):
        st.markdown(
            "- **PR-AUC:** precision–recall ranking summary; useful with rare positive outcomes.\n"
            "- **Brier score:** mean squared error of probability estimates; lower is better.\n"
            "- **Calibration:** compares predicted probabilities with observed outcome frequency.\n"
            "- The label is five-year bankruptcy, not observed contractual default.\n"
            "- The source has no observation dates or stable company IDs, so this is not a temporal backtest.\n"
            "- Performance estimates are uncertain because the held-out test set has few positive cases."
        )


def main() -> None:
    st.title("Credit Risk Assessment & Default Prediction System")
    profiles = ["Annual panel — chronological validation", "Polish baseline — random split"]
    default_profile = 0 if (ROOT / "models/temporal/credit_risk_model.joblib").exists() else 1
    profile = st.sidebar.selectbox("Dataset and validation", profiles, index=default_profile)
    if profile == profiles[0]:
        from temporal_views import render_temporal_dashboard
        render_temporal_dashboard(ROOT)
        return
    st.caption("Educational credit-risk analytics · public UCI source · local SQLite and scikit-learn model")
    missing = [path.name for path in (DATABASE_PATH, MODEL_PATH, PREDICTION_PATH) if not path.exists()]
    if missing:
        st.error("Required project artifacts are missing: " + ", ".join(missing))
        st.code("python scripts/load_sqlite.py\npython scripts/train_models.py")
        st.stop()

    try:
        data = load_dashboard_data()
        bundle = load_model_bundle()
    except Exception as error:
        st.error(f"Could not load dashboard data: {error}")
        st.stop()

    page = st.sidebar.radio(
        "Dashboard page",
        ["Portfolio Overview", "Company Credit Analysis", "Stress Testing", "Model Performance"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: {bundle['selected_model'].replace('_', ' ').title()}")
    st.sidebar.caption("Target proxy: bankruptcy within five years")
    if page == "Portfolio Overview":
        show_portfolio(data)
    elif page == "Company Credit Analysis":
        show_company_analysis(data, bundle)
    elif page == "Stress Testing":
        show_stress_testing(data, bundle)
    else:
        show_model_performance()


if __name__ == "__main__":
    main()
