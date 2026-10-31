"""Compare the reproduced results with the values printed in the manuscript. Prints PASS/FAIL for each value."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from common import TABLES

t5 = pd.read_csv(TABLES / 'Table5_activity_cv.csv').set_index('model')
t6 = pd.read_csv(TABLES / 'Table6_sex_cv.csv').set_index('model')
t7 = pd.read_csv(TABLES / 'Table7_mcnemar.csv')
te = pd.read_csv(TABLES / 'Table8_S1_S2_external.csv')
tr = pd.read_csv(TABLES / 'Section4_1_tracking.csv')
r = lambda x, d=1: float(round(100 * x, d))
checks = [
    # (description, reproduced, manuscript)
    ('Table 5 R3D-18 BA', r(t5.loc['R3D-18', 'balanced_accuracy']), 90.5),
    ('Table 5 R3D-18 CI', (r(t5.loc['R3D-18', 'ci_low']), r(t5.loc['R3D-18', 'ci_high'])), (86.2, 94.4)),
    ('Table 5 EfficientNet BA', r(t5.loc['EfficientNet-B0 + BiLSTM', 'balanced_accuracy']), 82.3),
    ('Table 5 Two-stream BA', r(t5.loc['Two-stream', 'balanced_accuracy']), 81.1),
    ('Table 5 Two-stream (res. aug.) BA', r(t5.loc['Two-stream (res. aug.)', 'balanced_accuracy']), 77.7),
    ('Table 5 R3D-18 waving / feeding recall', (r(t5.loc['R3D-18', 'recall_positive']), r(t5.loc['R3D-18', 'recall_negative'])), (88.7, 92.3)),
    ('Table 6 DINOv2 BA all / feeding', (r(t6.loc['DINOv2 + LR', 'all_balanced_accuracy']), r(t6.loc['DINOv2 + LR', 'feeding_balanced_accuracy'])), (94.8, 95.4)),
    ('Table 6 DINOv2 CI all', (r(t6.loc['DINOv2 + LR', 'all_ci_low']), r(t6.loc['DINOv2 + LR', 'all_ci_high'])), (91.4, 97.7)),
    ('Table 6 Two-stream (res. aug.) all / feeding', (r(t6.loc['Two-stream (res. aug.)', 'all_balanced_accuracy']), r(t6.loc['Two-stream (res. aug.)', 'feeding_balanced_accuracy'])), (70.4, 61.8)),
    ('Table 6 EfficientNet feeding CI', (r(t6.loc['EfficientNet-B0 + BiLSTM', 'feeding_ci_low']), r(t6.loc['EfficientNet-B0 + BiLSTM', 'feeding_ci_high'])), (85.0, 95.7)),
    ('Table 7 discordant counts', [tuple(int(v) for v in x) for x in t7[['a_only', 'b_only']].values],
     [(41, 12), (32, 11), (46, 13), (44, 48), (8, 16), (19, 6), (17, 7), (64, 4), (22, 13)]),
    ('Table 7 Holm p (sex, feeding: DINOv2 vs R3D-18)', float(round(t7.p_holm[5], 3)), 0.044),
    ('Table 8 sex E1 BA and CI', tuple(r(x) for x in te.loc[0, ['balanced_accuracy', 'ci_low', 'ci_high']]), (97.1, 92.6, 100.0)),
    ('Table 8 R3D-18 E1+E2 BA and CI', tuple(r(x) for x in te.loc[3, ['balanced_accuracy', 'ci_low', 'ci_high']]), (70.3, 51.8, 87.6)),
    ('Table S1 sex 90 clips BA and CI', tuple(r(x) for x in te.loc[6, ['balanced_accuracy', 'ci_low', 'ci_high']]), (96.5, 92.0, 100.0)),
    ('Table S1 R3D-18 sensitivity BA and CI', tuple(r(x) for x in te.loc[11, ['balanced_accuracy', 'ci_low', 'ci_high']]), (64.2, 39.3, 89.0)),
    ('Table S2 sex 64 clips BA and CI', tuple(r(x) for x in te.loc[12, ['balanced_accuracy', 'ci_low', 'ci_high']]), (90.6, 83.0, 96.9)),
    ('Section 4.1 training: direct / extrapolated / reused / single identity',
     tuple(float(v) for v in tr.loc[0, ['detected directly, %', 'extrapolated, %', 'last box reused, %', 'single track identity, %']]), (90.0, 5.9, 4.1, 83.0)),
    ('Section 4.1 external: direct / extrapolated / reused / single identity',
     tuple(float(v) for v in tr.loc[1, ['detected directly, %', 'extrapolated, %', 'last box reused, %', 'single track identity, %']]), (72.8, 13.1, 14.1, 88.9)),
]
fails = 0
for name, got, expected in checks:
    ok = got == expected
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: reproduced {got}" + ('' if ok else f'   manuscript {expected}'))
print(f'\n{len(checks) - fails} of {len(checks)} checks passed.')
sys.exit(1 if fails else 0)
