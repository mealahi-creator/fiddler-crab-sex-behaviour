"""Shared data loading and statistics for all analysis scripts.

Every number in the manuscript is computed from the saved per-sequence predictions in data/predictions/
with the functions below. Nothing is typed in by hand except the detector validation values
(data/detector/detector_comparison.csv), which come from the detector training report.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PRED = DATA / 'predictions'
RESULTS = ROOT / 'results'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# model names as used in the manuscript
R3D, EFF, TS, TSA, DINO = 'R3D-18', 'EfficientNet-B0 + BiLSTM', 'Two-stream', 'Two-stream (res. aug.)', 'DINOv2 + LR'
ACT_MODELS = [R3D, EFF, TS, TSA]
SEX_MODELS = [DINO, EFF, R3D, TSA, TS]
THRESHOLD = 0.5          # fixed in advance, never tuned
N_BOOT, BOOT_SEED = 2000, 0


# ------------------------------------------------------------------ metrics
def balanced_accuracy(t, p):
    """Mean of the two class recalls, with the positive class predicted when p > 0.5."""
    t, y = np.asarray(t), np.asarray(p) > THRESHOLD
    return 0.5 * (y[t == 1].mean() + (~y[t == 0]).mean())


def bootstrap_ci(t, p, video, B=N_BOOT, seed=BOOT_SEED, order='sorted'):
    """95% CI of balanced accuracy by resampling VIDEOS with replacement (cluster bootstrap, percentile method).

    order: how the list of videos is ordered before random draws. Both orders give equally valid intervals;
    they only change which random resamples are drawn. They are kept so that the numbers match the manuscript exactly:
    'sorted' (alphabetical) was used for cross-validation, 'appearance' (order in the file) by the locked external-test scripts.
    """
    t, p, video = np.asarray(t), np.asarray(p), np.asarray(video)
    rng = np.random.default_rng(seed)
    vids = np.unique(video) if order == 'sorted' else pd.unique(video)
    idx = {v: np.where(video == v)[0] for v in vids}
    out = []
    for _ in range(B):
        s = np.concatenate([idx[v] for v in rng.choice(vids, len(vids))])
        if 0 < t[s].sum() < len(s):          # both classes must be present
            out.append(balanced_accuracy(t[s], p[s]))
    return np.percentile(out, [2.5, 97.5])


def summary(t, p, video, order='sorted'):
    """All metrics reported in the result tables."""
    t, p = np.asarray(t), np.asarray(p)
    y = p > THRESHOLD
    lo, hi = bootstrap_ci(t, p, video, order=order)
    both = 0 < t.sum() < len(t)
    return dict(n=len(t), n_positive=int(t.sum()), balanced_accuracy=balanced_accuracy(t, p), ci_low=lo, ci_high=hi,
                recall_positive=y[t == 1].mean(), recall_negative=(~y[t == 0]).mean(), accuracy=(y == t).mean(),
                majority_baseline=max(t.mean(), 1 - t.mean()), roc_auc=roc_auc_score(t, p) if both else np.nan)


def mcnemar_exact(t, pa, pb):
    """Exact two-sided McNemar test = binomial test (p = 0.5) on the discordant sequences.
    Returns (A only correct, B only correct, p)."""
    t = np.asarray(t)
    a, b = (np.asarray(pa) > THRESHOLD) == t, (np.asarray(pb) > THRESHOLD) == t
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def holm(pvals):
    """Holm step-down adjustment: sort, multiply the i-th smallest by (m - i + 1), cap at 1, keep non-decreasing."""
    pvals = list(pvals)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}' if p < 0.1 else f'{p:.2f}'


def pct(x, d=1):
    return f'{100 * x:.{d}f}'


# ------------------------------------------------------------------ data
def _two_stream(prefix):
    """Average the two seeds of the two-stream network (one file per fold and seed)."""
    files = sorted((PRED / 'raw_twostream').glob(f'{prefix}_pred_fold*_seed*.csv'))
    assert len(files) == 10, f'expected 10 prediction files for {prefix}, found {len(files)}'
    d = pd.concat(pd.read_csv(f) for f in files)
    return d.groupby('seq')[['p_waving', 'p_male']].mean()


def load_cv():
    """One row per sequence (n = 278): labels, fold, video and the predicted probabilities of every model."""
    base = pd.read_csv(PRED / 'cv_r3d18.csv')
    cv = base[['seq', 'video', 'sex', 'activity', 'fold']].copy()
    sources = {R3D: base.set_index('seq'),
               EFF: pd.read_csv(PRED / 'cv_effb0_bilstm.csv').set_index('seq'),
               TS: _two_stream('noaug'), TSA: _two_stream('resaug')}
    for m, d in sources.items():
        cv[f'{m}|act'] = d.loc[cv.seq, 'p_waving'].values
        cv[f'{m}|sex'] = d.loc[cv.seq, 'p_male'].values
    enc = load_sex_variants()
    sel = enc[(enc.backbone == 'dinov2_s') & (enc.crop == 'full') & (enc.train_on == 'feeding_only')].set_index('seq')
    cv[f'{DINO}|sex'] = sel.loc[cv.seq, 'p_male_seq'].values
    cv['y_act'] = (cv.activity == 'waving').astype(int)
    cv['y_sex'] = (cv.sex == 'male').astype(int)
    cv['group'] = cv.sex + ' ' + cv.activity
    assert len(cv) == 278 and not cv.isna().any().any()
    return cv


def load_sex_variants():
    """Image-encoder sex classifiers: 2 encoders x 2 crops x 2 training sets, video-grouped CV."""
    return pd.read_csv(PRED / 'cv_sex_image_encoders.csv')


def load_external(name):
    """name: 'primary71' (pre-declared test, session E1), 'secondary90' (E1 + E2), 'mismatched64' (first run, Table S2)."""
    f = {'primary71': 'external_E1_primary_71clips.csv', 'secondary90': 'external_secondary_90clips.csv',
         'mismatched64': 'external_E1_mismatched_64clips.csv'}[name]
    d = pd.read_csv(PRED / f)
    d['blind_exclude'] = d.blind_exclude.astype(str).str.lower().eq('true')
    if 'low_fps' in d:
        d['low_fps'] = d.low_fps.astype(str).str.lower().eq('true')
    d['y_act'] = (d.activity == 'waving').astype(int)
    d['y_sex'] = (d.sex == 'male').astype(int)
    d['video'] = d.video.astype(str)          # bootstrap cluster = source video, as recorded by the locked scripts
    return d
