"""Run chronological development, locked test evaluation, and SQL persistence."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from credit_risk.ingestion.panel_loader import load_author_panel
from credit_risk.features.panel_features import build_panel_features
from credit_risk.models.temporal_training import evaluate_temporal
from credit_risk.database.panel_store import save_panel_database
from credit_risk.reporting.temporal_results import write_temporal_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/temporal_validation.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    panel, audit = load_author_panel(ROOT / 'data/raw/american_bankruptcy_2022.csv')
    features = build_panel_features(panel)
    bundle, predictions, card = evaluate_temporal(panel, features, config, audit, ROOT)
    save_panel_database(ROOT / 'data/processed/credit_risk_temporal.sqlite3', panel, features, predictions, card)
    write_temporal_results(ROOT)
    print(json.dumps({'run_id': card['run_id'], 'selected_model': card['selected_model'],
                      'probability_source': card['probability_source'], 'held_out_metrics': card['test_metrics']}, indent=2))
    print('Saved separate temporal artifacts, SQLite data, and docs/temporal_results.md.')

if __name__ == '__main__':
    main()
