"""Dedicated normalized SQLite snapshot for annual financial records and predictions."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pandas as pd
from credit_risk.ingestion.panel_loader import FIELD_MAP
from credit_risk.features.panel_features import FEATURE_COLUMNS
from credit_risk.models.temporal_model import PANEL_GRADE_BANDS


def save_panel_database(path, panel, features, predictions, card):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building')
    if temporary.exists():
        temporary.unlink()
    raw_columns = list(FIELD_MAP.values())
    with sqlite3.connect(temporary) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.executescript(f'''
        CREATE TABLE companies (company_id TEXT PRIMARY KEY);
        CREATE TABLE financial_statements (
            observation_id TEXT PRIMARY KEY, company_id TEXT NOT NULL REFERENCES companies(company_id),
            fiscal_year INTEGER NOT NULL, {', '.join(name + ' REAL' for name in raw_columns)},
            UNIQUE(company_id, fiscal_year));
        CREATE TABLE financial_ratios (
            observation_id TEXT PRIMARY KEY REFERENCES financial_statements(observation_id),
            {', '.join(name + ' REAL' for name in FEATURE_COLUMNS)});
        CREATE TABLE model_runs (run_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL);
        CREATE TABLE risk_grades (grade TEXT PRIMARY KEY, lower_pd REAL, upper_pd_exclusive REAL);
        CREATE TABLE model_predictions (
            observation_id TEXT PRIMARY KEY REFERENCES financial_statements(observation_id),
            run_id TEXT NOT NULL REFERENCES model_runs(run_id), selected_pd REAL CHECK(selected_pd BETWEEN 0 AND 1),
            raw_pd REAL, calibrated_pd REAL, internal_grade TEXT REFERENCES risk_grades(grade),
            risk_score INTEGER, bankruptcy_next_year INTEGER, cohort TEXT,
            missing_features INTEGER, outside_training_range INTEGER);
        CREATE INDEX statement_year ON financial_statements(fiscal_year);
        CREATE INDEX prediction_grade ON model_predictions(internal_grade);
        ''')
        connection.executemany('INSERT INTO companies VALUES (?)', [(str(x),) for x in panel.company_id.unique()])
        raw = panel[['company_id', 'fiscal_year', *raw_columns]].copy()
        ids = panel.company_id + ':' + panel.fiscal_year.astype(str)
        raw.insert(0, 'observation_id', ids)
        raw = raw.astype(object).where(raw.notna(), None)
        connection.executemany(f"INSERT INTO financial_statements VALUES ({','.join('?' for _ in raw.columns)})",
                               raw.itertuples(index=False, name=None))
        ratios = features.copy()
        ratios.insert(0, 'observation_id', ids)
        ratios = ratios.astype(object).where(ratios.notna(), None)
        connection.executemany(f"INSERT INTO financial_ratios VALUES ({','.join('?' for _ in ratios.columns)})",
                               ratios.itertuples(index=False, name=None))
        connection.execute('INSERT INTO model_runs VALUES (?, ?)', (card['run_id'], json.dumps(card)))
        lower = 0.
        for upper, grade in PANEL_GRADE_BANDS:
            connection.execute('INSERT INTO risk_grades VALUES (?, ?, ?)', (grade, lower, upper))
            lower = upper
        columns = ['observation_id', 'selected_pd', 'raw_pd', 'calibrated_pd', 'internal_grade',
                   'risk_score', 'bankruptcy_next_year', 'cohort', 'missing_features', 'outside_training_range']
        rows = predictions[columns].copy()
        rows.insert(1, 'run_id', card['run_id'])
        connection.executemany(f"INSERT INTO model_predictions VALUES ({','.join('?' for _ in rows.columns)})",
                               rows.itertuples(index=False, name=None))
        connection.commit()
    temporary.replace(path)


def read_panel_predictions(path):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as connection:
        return pd.read_sql_query('''SELECT p.*, s.company_id, s.fiscal_year
            FROM model_predictions p JOIN financial_statements s USING(observation_id)
            ORDER BY s.fiscal_year, s.company_id''', connection)


def read_company_history(path, company_id):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as connection:
        return pd.read_sql_query('SELECT * FROM financial_statements WHERE company_id = ? ORDER BY fiscal_year',
                                 connection, params=[company_id])
