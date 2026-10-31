# =====================================================================
# EXTERNAL ACTIVITY TEST with the LOCKED R3D-18 (Amendment 2 primary model)
#
# Uses exactly the same 90 clips (and blind-QC flags) as the 2026-10-14 run
# of external_test_v2_once_colab.py, so results can be compared clip by clip.
# Checks both .pt files against LOCK_r3d18.json before predicting.
# Model input identical to R3D-18 training: 24 evenly spaced frames, RGB,
# resize 224, Kinetics mean/std, trilinear to 24x112x112; mean p(waving)
# of the 2 seeds; waving if > 0.5.  Run ONCE (refuses to overwrite).
# Colab: GPU runtime is faster but CPU also works (90 clips).
# =====================================================================
import os, re, json, hashlib, datetime, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from google.colab import drive
drive.mount('/content/drive')

MODELS = '/content/drive/MyDrive/Crab_Final_October/'
R3D    = MODELS + 'r3d18_locked/'
LOCK   = MODELS + 'LOCK_r3d18.json'
DATA   = '/content/drive/MyDrive/Crab_External_v2/Processed_dataset/'
CLIPS  = MODELS + 'external_test_v2/external_v2_predictions_LOCKED.csv'   # 90-clip run of 2026-10-14
OUT    = MODELS + 'external_r3d18/'
LOW_FPS = ['VID_20250803_100415']   # real frame rate 17.7 fps < 25 -> excluded by the STEP 0 rule
os.makedirs(OUT, exist_ok=True)
PRED = OUT + 'external_r3d18_predictions.csv'
assert not os.path.exists(PRED), 'R3D-18 external test already run. Do not re-run; send me the files in ' + OUT
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 1. locked weights ----------
lock = json.load(open(LOCK))
for f, info in lock['files'].items():
    h = hashlib.sha256(open(R3D + f, 'rb').read()).hexdigest()
    assert h == info['sha256'], f'{f} changed since lock!'
print('Both R3D-18 files match LOCK_r3d18.json (locked', lock['locked_utc'] + ')')

class R3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torchvision.models.video.r3d_18(weights=None); self.b.fc = nn.Identity()
        self.drop = nn.Dropout(0.5); self.act = nn.Linear(512, 2); self.sex = nn.Linear(512, 2)
    def forward(self, x):
        h = self.drop(self.b(x)); return self.act(h), self.sex(h)

# ---------- 2. same clips as the two-stream run ----------
ext = pd.read_csv(CLIPS)
ext['path'] = DATA + ext.sex + '/' + ext.activity + '/' + ext.seq_name
miss = [p for p in ext.path if not os.path.isdir(p)]
assert not miss, f'{len(miss)} clip folders not found, e.g. {miss[:3]}'
ext['source'] = np.where(ext.seq_name.str.startswith('VID_'), 'Aug2025_phone', 'Feb2026_session')
ext['low_fps'] = ext.seq_name.str.startswith(tuple(LOW_FPS))
print(len(ext), 'clips:', ext.groupby(['source', 'sex', 'activity']).size().to_dict())

KIN_M, KIN_S = torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989])
def load_clip(d):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
    f = [f[i] for i in np.linspace(0, len(f) - 1, 24).astype(int)]
    x = torch.from_numpy(np.stack([np.array(Image.open(p).convert('RGB').resize((224, 224))) for p in f])).float().div(255)
    x = ((x - KIN_M) / KIN_S).permute(3, 0, 1, 2)[None]                       # 1,3,T,H,W
    return F.interpolate(x, size=(24, 112, 112), mode='trilinear', align_corners=False)

P = []
for f in lock['files']:
    m = R3DNet(); m.load_state_dict(torch.load(R3D + f, map_location='cpu')); m.eval().to(DEV)
    with torch.no_grad():
        P.append([m(load_clip(d).to(DEV))[0].float().softmax(1)[0, 1].item() for d in ext.path])
    del m
ext['p_waving_r3d18'] = np.mean(P, 0)
ext.drop(columns='path').to_csv(PRED, index=False)
print('Saved', PRED)

# ---------- 3. metrics (same definitions as before) ----------
def metrics(t, p):
    yhat = p > .5; pos, neg = t == 1, t == 0
    r1 = yhat[pos].mean() if pos.any() else np.nan; r0 = (~yhat[neg]).mean() if neg.any() else np.nan
    return dict(n=len(t), n_pos=int(pos.sum()), acc=(yhat == t).mean(), bal=np.nanmean([r1, r0]),
                recall_waving=r1, recall_feeding=r0, majority=max(pos.mean(), neg.mean()),
                auc=roc_auc_score(t, p) if 0 < pos.sum() < len(t) else np.nan)
def boot_ci(df, pcol, B=2000):
    rng = np.random.default_rng(0); v = df.video.unique(); g = {k: x for k, x in df.groupby('video')}; out = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(v, len(v))]); t = s.y.values
        if 0 < t.sum() < len(t): out.append(metrics(t, s[pcol].values)['bal'])
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

ext['y'] = (ext.activity == 'waving').astype(int)
sets = [('A_all', ext.index == ext.index),
        ('B_blindQC', ~ext.blind_exclude.astype(bool)),
        ('C_blindQC_and_fps_ok', ~ext.blind_exclude.astype(bool) & ~ext.low_fps)]
rows = []
for name, keep in sets:
    d = ext[keep]
    for model, pcol in [('R3D-18 (primary)', 'p_waving_r3d18'), ('two-stream aug', 'p_waving_aug'), ('two-stream no-aug', 'p_waving_noaug')]:
        lo, hi = boot_ci(d, pcol)
        rows.append(dict(set=name, model=model, **metrics(d.y.values, d[pcol].values), bal_ci_lo=lo, bal_ci_hi=hi))
    a = (d.p_waving_r3d18 > .5) == d.y; b = (d.p_waving_aug > .5) == d.y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    rows.append(dict(set=name, model=f'McNemar R3D-18 vs two-stream aug: {n01} vs {n10}',
                     n=len(d), acc=binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0))
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT + 'external_r3d18_metrics.csv', index=False)
json.dump({'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           'lock_utc': lock['locked_utc'], 'clips_from': CLIPS, 'n': len(ext), 'low_fps_rule': LOW_FPS,
           'note': 'McNemar rows: the p-value is stored in the acc column'},
          open(OUT + 'run_info.json', 'w'), indent=2)
pd.set_option('display.width', 220); print(res.to_string(index=False))
print('\nWaving clips:'); print(ext[ext.y == 1][['seq_name', 'source', 'p_waving_r3d18', 'p_waving_aug']].round(3).to_string(index=False))
print('\nDone. Send me the 3 files in', OUT)
