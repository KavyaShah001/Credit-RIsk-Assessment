PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    company_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_row_number INTEGER NOT NULL,
    company_name TEXT,
    sector TEXT,
    UNIQUE(source_dataset, source_row_number)
);

CREATE TABLE IF NOT EXISTS financial_observations (
    observation_id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL UNIQUE REFERENCES companies(company_id) ON DELETE CASCADE,
    observation_case TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratio_dictionary (
    ratio_code TEXT PRIMARY KEY,
    source_attribute TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_ratios (
    observation_id INTEGER NOT NULL REFERENCES financial_observations(observation_id) ON DELETE CASCADE,
    ratio_code TEXT NOT NULL REFERENCES ratio_dictionary(ratio_code),
    ratio_value REAL,
    PRIMARY KEY(observation_id, ratio_code)
);

CREATE TABLE IF NOT EXISTS default_outcomes (
    company_id TEXT PRIMARY KEY REFERENCES companies(company_id) ON DELETE CASCADE,
    outcome_name TEXT NOT NULL,
    horizon_years INTEGER NOT NULL,
    defaulted INTEGER NOT NULL CHECK(defaulted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_dataset TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_license TEXT NOT NULL,
    rows_loaded INTEGER NOT NULL,
    defaults_loaded INTEGER NOT NULL,
    missing_ratio_cells INTEGER NOT NULL,
    loaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_outcomes_defaulted ON default_outcomes(defaulted);
CREATE INDEX IF NOT EXISTS idx_ratios_code_value ON financial_ratios(ratio_code, ratio_value);
