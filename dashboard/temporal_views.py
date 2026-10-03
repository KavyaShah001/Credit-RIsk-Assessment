"""Calendar-year validation and reusable annual-statement analysis views."""
from __future__ import annotations
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from credit_risk.database.panel_store import read_panel_predictions, read_company_history
from credit_risk.features.panel_features import build_panel_features, FEATURE_COLUMNS, RATIO_FORMULAS
from credit_risk.ingestion.panel_loader import validate_panel, FIELD_MAP
from credit_risk.models.temporal_model import score_features, PANEL_GRADE_BANDS
from credit_risk.explainability.local_sensitivity import local_ratio_sensitivity
from credit_risk.reporting.panel_report import render_panel_report
from credit_risk.stress_testing.statement_scenario import apply_margin_compression


@st.cache_resource
def _bundle(path, version):
    return joblib.load(path)


@st.cache_data
def _predictions(path, version):
    return read_panel_predictions(path)


@st.cache_data
def _history(path, version, identifier):
    return read_company_history(path, identifier)


def _drivers(bundle, feature_row):
    local = local_ratio_sensitivity(bundle['model'], feature_row, bundle['training_medians'], FEATURE_COLUMNS)
    local['description'] = local.feature.map(bundle['feature_descriptions'])
    local['risk_effect_pp'] = -100 * local.pd_change
    return local[['feature', 'description', 'observed_value', 'training_median', 'risk_effect_pp']]


def _selected_observation(predictions, database, version, prefix):
    year = st.selectbox('Fiscal year', sorted(predictions.fiscal_year.unique(), reverse=True), key=prefix + '_year')
    eligible = predictions[predictions.fiscal_year.eq(year)]
    identifier = st.selectbox('Anonymous company ID', eligible.company_id.tolist(), key=prefix + '_entity')
    row = eligible[eligible.company_id.eq(identifier)].iloc[0]
    history = _history(str(database), version, identifier)
    # Explicitly restrict features and displayed history to information no later than the selected year.
    history = history[history.fiscal_year.le(year)].copy()
    feature_row = build_panel_features(history).loc[history.fiscal_year.eq(year)]
    return row, history, feature_row


def _portfolio(predictions):
    st.header('Annual portfolio overview')
    st.caption('Choose one fiscal-year snapshot. Each anonymous company appears once in that snapshot.')
    year = st.selectbox('Portfolio fiscal year', sorted(predictions.fiscal_year.unique(), reverse=True))
    view = predictions[predictions.fiscal_year.eq(year)]
    selected = st.multiselect('Select companies (empty means all)', view.company_id.tolist())
    if selected:
        view = view[view.company_id.isin(selected)]
    a, b, c, d = st.columns(4)
    a.metric('Companies', f'{len(view):,}')
    b.metric('Average annual PD proxy', f'{view.selected_pd.mean():.2%}')
    c.metric('Median annual PD proxy', f'{view.selected_pd.median():.2%}')
    d.metric('Reserved unseen companies', int(view.cohort.eq('reserved_unseen_entities').sum()))
    counts = view.internal_grade.value_counts().reindex([grade for _, grade in PANEL_GRADE_BANDS], fill_value=0)
    st.bar_chart(counts.rename('Companies'))
    st.caption('T1 <0.25%; T2 0.25–<1%; T3 1–<3%; T4 3–<10%; T5 ≥10%. Project Internal Credit Risk Rating; one-year proxy bands.')
    table = view.sort_values('selected_pd', ascending=False)[['company_id', 'fiscal_year', 'selected_pd', 'internal_grade', 'risk_score', 'cohort']]
    st.dataframe(table, hide_index=True, width='stretch', column_config={'selected_pd': st.column_config.NumberColumn('PD proxy', format='percent')})
    st.info('Exposure, LGD and sector are unavailable in this source release. Portfolio summaries use equal company weights.')


def _company(predictions, database, version, bundle, card):
    st.header('Company history and credit analysis')
    row, history, feature_row = _selected_observation(predictions, database, version, 'analysis')
    a, b, c = st.columns(3)
    a.metric('Next-fiscal-year bankruptcy proxy', f'{row.selected_pd:.2%}')
    b.metric('Project internal grade', row.internal_grade)
    c.metric('Risk score', f'{row.risk_score}/100')
    st.caption(f'Observation cohort: {row.cohort}. IDs persist across annual observations but do not reveal company names.')
    ratio_history = build_panel_features(history)
    ratio_history['fiscal_year'] = history.fiscal_year
    st.subheader('Historical financial indicators')
    st.line_chart(ratio_history.set_index('fiscal_year')[['current_ratio', 'liabilities_to_assets', 'ebitda_margin']])
    current_ratios = feature_row.iloc[0][list(RATIO_FORMULAS)]
    st.dataframe(pd.DataFrame({'feature': current_ratios.index, 'value': current_ratios.values,
                              'formula': [RATIO_FORMULAS[name] for name in current_ratios.index]}), hide_index=True, width='stretch')
    drivers = _drivers(bundle, feature_row)
    st.subheader('Drivers relative to training medians')
    st.caption('Positive values associate with higher model PD; negative values with lower model PD. Effects are sensitivity checks and do not sum to PD.')
    st.dataframe(drivers.head(10), hide_index=True, width='stretch')
    with st.expander('Annual statement fields as supplied'):
        st.dataframe(history[['fiscal_year', *FIELD_MAP.values()]], hide_index=True, width='stretch')
    html = render_panel_report(row.company_id, row.fiscal_year, row, feature_row.iloc[0], drivers.head(10), card)
    st.download_button('Download annual credit-risk report', html, f'credit-risk-{row.company_id}-{row.fiscal_year}.html', 'text/html')


def _validation(root, card):
    reports = root / 'reports/temporal'
    st.header('Time-based validation and generalization evidence')
    st.warning('Fiscal-year backtest with reconstructed next-year outcomes. Exact filing dates and statement publication timestamps are unavailable.')
    st.subheader('Chronological data separation')
    st.dataframe(pd.read_csv(reports / 'split_summary.csv'), hide_index=True, width='stretch')
    st.caption('A full fiscal year separates each stage so prior outcome windows mature before the next stage. Reserved company IDs never enter development.')
    st.subheader('Model comparison — validation period only')
    comparison = pd.read_csv(reports / 'validation_comparison.csv')
    st.dataframe(comparison[['model', 'roc_auc', 'pr_auc', 'brier', 'calibrated_brier', 'use_calibrated']], hide_index=True, width='stretch')
    st.subheader('Final held-out results and baseline comparison')
    metrics = pd.read_csv(reports / 'test_metrics.csv')
    st.dataframe(metrics, hide_index=True, width='stretch')
    selected = metrics[metrics.probability_source.eq('selected')].iloc[0]
    baseline = metrics[metrics.probability_source.eq('training_prevalence_baseline')].iloc[0]
    if selected.brier >= baseline.brier:
        st.warning('The selected model does not improve Brier score over the training-prevalence baseline on this final sample.')
    else:
        st.caption(f"Brier improvement relative to the training-prevalence baseline: {1 - selected.brier / baseline.brier:.1%}.")
    a, b = st.columns(2)
    with a:
        st.subheader('Annual performance')
        st.dataframe(pd.read_csv(reports / 'test_by_year.csv')[['fiscal_year', 'rows', 'events', 'roc_auc', 'pr_auc', 'brier']], hide_index=True)
    with b:
        st.subheader('Unseen-company performance')
        st.dataframe(pd.read_csv(reports / 'test_by_cohort.csv')[['cohort', 'companies', 'events', 'roc_auc', 'pr_auc', 'brier']], hide_index=True)
    st.subheader('95% uncertainty intervals')
    st.dataframe(pd.read_csv(reports / 'bootstrap_intervals.csv'), hide_index=True, width='stretch')
    st.caption('Whole-company bootstrap preserves dependence across annual records. These intervals exclude uncertainty about labels, model selection and future economic regimes.')
    a, b = st.columns(2)
    with a:
        st.subheader('ROC curve — final test')
        st.line_chart(pd.read_csv(reports / 'roc.csv').set_index('false_positive_rate'))
    with b:
        st.subheader('Precision–recall curve — final test')
        st.line_chart(pd.read_csv(reports / 'precision_recall.csv').set_index('recall'))
    a, b = st.columns(2)
    with a:
        st.subheader('Calibration — selected probabilities')
        calibration = pd.read_csv(reports / 'calibration_selected.csv')
        points = calibration.set_index('mean_predicted_pd')[['observed_rate']]
        points['perfect_calibration'] = points.index
        st.line_chart(points)
        st.dataframe(calibration, hide_index=True)
    with b:
        st.subheader('Confusion matrix at validation threshold')
        st.caption(f"Threshold fixed before the test: {card['threshold']:.4f}")
        st.dataframe(pd.DataFrame([[int(selected.tn), int(selected.fp)], [int(selected.fn), int(selected.tp)]],
                                  index=['Actual no bankruptcy', 'Actual bankruptcy'], columns=['Predicted no bankruptcy', 'Predicted bankruptcy']))
    st.subheader('Earlier walk-forward diagnostics')
    st.dataframe(pd.read_csv(reports / 'walk_forward.csv'), hide_index=True, width='stretch')
    st.caption('Each earlier fold repeats training, later calibration, later model selection, and a subsequent evaluation year. Final test years are excluded.')
    st.subheader('Feature distribution shift')
    st.dataframe(pd.read_csv(reports / 'feature_drift.csv').head(12), hide_index=True, width='stretch')
    st.caption('PSI uses training-defined bins and a missing-value bin. It is a drift diagnostic, not proof of model failure or success.')
    with st.expander('Feature importance, source audit and limitations'):
        st.dataframe(pd.read_csv(reports / 'feature_importance.csv'), hide_index=True)
        st.json(card['source_audit'])
        for item in card['limitations']:
            st.write('• ' + item)


def _stress(predictions, database, version, bundle, card):
    st.header('Operating-cost stress scenario')
    row, history, feature_row = _selected_observation(predictions, database, version, 'stress')
    compression = st.slider('EBITDA margin compression (percentage points)', 0., 20., 3., .5)
    st.caption('Assumption: extra operating costs equal revenue × selected margin reduction. EBITDA, EBIT and net income fall by that amount; no tax relief is assumed. This partial income-statement scenario does not project cash, financing or balance-sheet changes.')
    try:
        stressed, extra_cost = apply_margin_compression(history, int(row.fiscal_year), compression)
    except ValueError as error:
        st.info(str(error))
        return
    changed_features = build_panel_features(stressed).loc[stressed.fiscal_year.eq(row.fiscal_year)]
    risk = score_features(bundle, changed_features).iloc[0]
    change = float(risk.selected_pd - row.selected_pd)
    a, b, c = st.columns(3)
    a.metric('Base annual PD proxy', f'{row.selected_pd:.2%}')
    b.metric('Scenario annual PD proxy', f'{risk.selected_pd:.2%}')
    c.metric('Change', f'{change * 100:+.2f} pp')
    st.write(f'Grade {row.internal_grade} → {risk.internal_grade}. Predictions need not respond monotonically to a scenario.')
    comparison = pd.DataFrame({'base': feature_row.iloc[0], 'scenario': changed_features.iloc[0]})
    st.dataframe(comparison.loc[['ebitda_margin', 'ebit_margin', 'net_margin', 'return_on_assets', 'long_term_debt_to_ebitda']])
    drivers = _drivers(bundle, feature_row)
    scenario = f'Additional operating costs {extra_cost:.5g} in source monetary units; margin compression {compression:.1f} pp. Base PD {row.selected_pd:.2%}; scenario PD {risk.selected_pd:.2%}; change {change*100:+.2f} pp. Taxes, financing and balance-sheet feedback are not modeled.'
    html = render_panel_report(row.company_id, row.fiscal_year, row, feature_row.iloc[0], drivers.head(10), card, scenario)
    st.download_button('Download report with scenario', html, f'scenario-{row.company_id}-{row.fiscal_year}.html', 'text/html')


def _score_csv(bundle):
    st.header('Score additional annual statement data')
    st.write('Use the canonical CSV fields below. Include earlier years for each company when available; annual lags are only calculated for consecutive years.')
    template = pd.DataFrame(columns=['company_id', 'fiscal_year', *FIELD_MAP.values()]).to_csv(index=False)
    st.download_button('Download empty CSV schema', template, 'annual_statement_schema.csv', 'text/csv')
    st.info('All amounts within a row must use a consistent currency and scale. Missing values may be blank. Dates and company IDs must be real source identifiers. A successful score does not establish validity for a new population.')
    uploaded = st.file_uploader('Annual statement CSV', type=['csv'])
    if uploaded is None:
        return
    try:
        panel = validate_panel(pd.read_csv(io.BytesIO(uploaded.getvalue())))
        features = build_panel_features(panel)
        predictions = pd.concat([panel[['company_id', 'fiscal_year']], score_features(bundle, features)], axis=1)
    except (ValueError, TypeError, KeyError) as error:
        st.error(f'Input data rejected: {error}')
        return
    st.dataframe(predictions, hide_index=True, width='stretch')
    st.caption('outside_training_range counts features outside training 0.5–99.5 percentile bounds; missing_features counts features imputed for prediction. Review these diagnostics before interpreting results.')
    st.download_button('Download scored observations', predictions.to_csv(index=False), 'annual_credit_scores.csv', 'text/csv')


def render_temporal_dashboard(root):
    root = Path(root)
    model = root / 'models/temporal/credit_risk_model.joblib'
    database = root / 'data/processed/credit_risk_temporal.sqlite3'
    manifest = root / 'reports/temporal/model_card.json'
    if not all(path.exists() for path in [model, database, manifest]):
        st.info('Prepare and train the annual panel to activate chronological validation.')
        st.code('.venv/bin/python scripts/prepare_temporal_dataset.py\n.venv/bin/python scripts/run_temporal_pipeline.py')
        return
    bundle = _bundle(str(model), model.stat().st_mtime_ns)
    predictions = _predictions(str(database), database.stat().st_mtime_ns)
    card = json.loads(manifest.read_text())
    if set(predictions.run_id) != {bundle['run_id']} or card['run_id'] != bundle['run_id']:
        st.error('Data and model artifacts refer to different runs. Finish the pipeline before reviewing the dashboard.')
        return
    st.caption('Annual financial statements · chronological validation · persistent anonymous entities')
    page = st.sidebar.radio('Dashboard page', ['Portfolio Overview', 'Company Credit Analysis', 'Stress Testing', 'Temporal Validation', 'Score New CSV'])
    st.sidebar.caption('Target: reconstructed next-fiscal-year bankruptcy proxy')
    st.sidebar.caption(f"Selected model: {bundle['selected_model'].replace('_', ' ')}")
    st.sidebar.caption('Annual grades T1–T5 differ from the legacy five-year grades.')
    if page == 'Portfolio Overview':
        _portfolio(predictions)
    elif page == 'Company Credit Analysis':
        _company(predictions, database, database.stat().st_mtime_ns, bundle, card)
    elif page == 'Temporal Validation':
        _validation(root, card)
    elif page == 'Stress Testing':
        _stress(predictions, database, database.stat().st_mtime_ns, bundle, card)
    else:
        _score_csv(bundle)
