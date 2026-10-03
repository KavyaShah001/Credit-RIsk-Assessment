"""Score new canonical annual statements without requiring outcome labels."""
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import pandas as pd
from credit_risk.ingestion.panel_loader import validate_panel
from credit_risk.features.panel_features import build_panel_features
from credit_risk.models.temporal_model import score_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/scored_panel.csv')
    args = parser.parse_args()
    panel = validate_panel(pd.read_csv(args.csv_path))
    features = build_panel_features(panel)
    bundle = joblib.load(ROOT / 'models/temporal/credit_risk_model.joblib')
    results = pd.concat([panel[['company_id', 'fiscal_year']], score_features(bundle, features)], axis=1)
    results['model_run_id'] = bundle['run_id']
    results['outcome'] = bundle['target']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f'Scored {len(results):,} observations; saved {args.output}')
    print('Read missing_features and outside_training_range. New populations remain unvalidated.')

if __name__ == '__main__':
    main()
