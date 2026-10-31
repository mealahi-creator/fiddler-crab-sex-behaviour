"""Section 4.1: detection and tracking continuity, computed from the complete tracking logs.

state = 'tracked'      crab detected directly by YOLOv11 + BoT-SORT
        'interpolated' box extrapolated from the track's last position and velocity (gaps of up to 5 frames)
        'recovered'    last box of the most confident track reused
Output: results/tables/Section4_1_tracking.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from common import DATA, TABLES, load_cv

cv = load_cv()
keep = cv[['seq', 'sex', 'activity']].rename(columns={'seq': 'sequence'})
rows = []
for label, f, subset in [('Training sequences used in the study', 'tracking_log_training.csv', keep),
                         ('External clips (E1 + E2)', 'tracking_log_external.csv', None)]:
    t = pd.read_csv(DATA / 'tracking' / f)
    if subset is not None:                       # the log also covers sequences removed before the final dataset
        t = t.merge(subset, on=['sequence', 'sex', 'activity'])
    share = t.state.value_counts(normalize=True)
    ids = t.groupby(['sequence', 'sex', 'activity']).track_id.nunique()
    rows.append({'set': label, 'sequences': len(ids), 'frames': len(t),
                 'detected directly, %': round(100 * share.get('tracked', 0), 1),
                 'extrapolated, %': round(100 * share.get('interpolated', 0), 1),
                 'last box reused, %': round(100 * share.get('recovered', 0), 1),
                 'mean confidence (direct detections)': round(t[t.state == 'tracked'].confidence.mean(), 2),
                 'single track identity, %': round(100 * (ids == 1).mean(), 1),
                 'mean identities per sequence': round(ids.mean(), 2)})
res = pd.DataFrame(rows)
res.to_csv(TABLES / 'Section4_1_tracking.csv', index=False)
print(res.T.to_string(header=False))
print(f'\n{len(cv) - rows[0]["sequences"]} of {len(cv)} sequences have no tracking record (a two-frame sequence).')
