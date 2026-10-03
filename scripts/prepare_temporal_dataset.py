"""Download the pinned author source atomically, then audit its labeling contract."""
import json
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from credit_risk.ingestion.panel_loader import SOURCE_URL, SOURCE_SHA256, load_author_panel


def main():
    raw = ROOT / 'data/raw/american_bankruptcy_2022.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'credit-risk-education-project/2.0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError('Downloaded source checksum does not match the pinned release.')
        temporary = raw.with_suffix('.download')
        temporary.write_bytes(content)
        temporary.replace(raw)
    panel, audit = load_author_panel(raw)
    (ROOT / 'reports/temporal').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    panel.to_csv(ROOT / 'data/processed/annual_financial_panel.csv', index=False)
    (ROOT / 'reports/temporal/data_audit.json').write_text(json.dumps(audit, indent=2))
    print(f"Audited {len(panel):,} annual observations, {audit['companies']:,} anonymous entities, {audit['reconstructed_events']} reconstructed events.")
    print('All 20 annual event counts match the dataset paper. Exact filing dates remain unavailable.')

if __name__ == '__main__':
    main()
