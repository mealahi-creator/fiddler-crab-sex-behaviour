# =====================================================================
# EXTERNAL TEST — run ONCE with the locked models. No tuning, no re-runs.
#
# Colab, GPU (CPU also works, ~10 min). Upload to /content/:
#   LOCKED_external_exclusions_scorer1.csv
#   external_seq_to_video.csv
# Needs on Drive (already there):
#   /content/drive/MyDrive/Crab_Final/            (locked models + LOCK.json)
#   /content/drive/MyDrive/External_Validation_Data_CLEAN/   (the 64 sequences)
#
# Pre-declared (written to ANALYSIS_PLAN.json BEFORE any prediction is made):
#   Sets:  A = all 64 | B = blind QC as scored (41) | C = strict criteria (45)
#   Activity: primary = aug (mean of 2 seeds), secondary = no-aug; waving if p > 0.5
#   Sex: DINOv2 + logistic regression; male if p > 0.5
#   Metrics: accuracy, balanced accuracy, per-class recall, majority baseline,
#            AUC, 95% CI by bootstrapping videos (2000 resamples, seed 0)
#   Preprocessing identical to training: 24 evenly spaced frames, resize 224,
#   NO blur, NO resolution matching, NO threshold change.
# =====================================================================
import os, json, glob, hashlib, datetime, numpy as np, pandas as pd, cv2, tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')

MODELS = '/content/drive/MyDrive/Crab_Final/'
EXT    = '/content/drive/MyDrive/External_Validation_Data_CLEAN/'
OUT    = MODELS + 'external_test/'
os.makedirs(OUT, exist_ok=True)
PRED_FILE = OUT + 'external_predictions_LOCKED.csv'
assert not os.path.exists(PRED_FILE), 'External test already run once. Do not re-run; send me the existing file.'

# ---------- 1. verify the models are exactly the locked ones ----------
lock = json.load(open(MODELS + 'LOCK.json'))
for f, h in lock['sha256'].items():
    assert hashlib.sha256(open(MODELS + f, 'rb').read()).hexdigest() == h, f'{f} changed since lock!'
print('All 6 model files match LOCK.json')

# ---------- 2. sequences, exclusions, plan ----------
qc  = pd.read_csv('/content/LOCKED_external_exclusions_scorer1.csv')
vid = pd.read_csv('/content/external_seq_to_video.csv')[['seq_name', 'video']]
ext = qc[['seq_name', 'qc_id', 'sex', 'activity', 'blind_exclude', 'strict_exclude']].merge(vid, on='seq_name')
assert len(ext) == 64
dirs = {os.path.basename(d): d for d in glob.glob(EXT + '**/seq_*', recursive=True) if os.path.isdir(d)}
ext['path'] = ext.seq_name.map(dirs)
assert ext.path.notna().all(), ext[ext.path.isna()].seq_name.tolist()

plan = {'written_utc': datetime.datetime.utcnow().isoformat(timespec='seconds'),
        'lock_utc': lock['locked_at_utc'],
        'sets': {'A_all': 64, 'B_blindQC': int((~ext.blind_exclude).sum()), 'C_strict': int((~ext.strict_exclude).sum())},
        'excluded_B': sorted(ext.seq_name[ext.blind_exclude]), 'excluded_C': sorted(ext.seq_name[ext.strict_exclude]),
        'thresholds': 0.5, 'preprocessing': '24 frames, resize 224, no blur/resolution matching',
        'metrics': 'acc, balanced acc, per-class recall, majority baseline, AUC, video-bootstrap 95% CI (2000, seed 0)'}
json.dump(plan, open(OUT + 'ANALYSIS_PLAN.json', 'w'), indent=2)
print('Plan written:', plan['sets'])

# ---------- 3. activity predictions (identical preprocessing to training) ----------
def frames(d, n):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
    return [f[i] for i in np.linspace(0, len(f) - 1, n).astype(int)]

def two_stream_inputs(d):
    rgb, dif, prev = [], [], None
    for p in frames(d, 24):
        f = cv2.resize(cv2.imread(p), (224, 224)); g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        rgb.append(f.astype(np.float32) / 255.0)
        if prev is None: prev = g
        dif.append((cv2.absdiff(g, prev).astype(np.float32) / 255.0)[..., None]); prev = g
    return np.array(rgb)[None], np.array(dif)[None]

X = [two_stream_inputs(d) for d in ext.path]
for variant in ['aug', 'noaug']:
    P = []
    for seed in lock['activity']['seeds']:
        m = tf.keras.models.load_model(MODELS + f'activity_{variant}_seed{seed}.keras', compile=False)
        p = []
        for r, df_ in X:
            pr = m.predict({'rgb_input': r, 'diff_input': df_}, verbose=0)
            p.append(float((pr['activity_output'] if isinstance(pr, dict) else pr[0])[0, 1]))
        P.append(p); tf.keras.backend.clear_session()
    ext[f'p_waving_{variant}'] = np.mean(P, 0)

# ---------- 4. sex predictions ----------
import torch, torchvision.transforms as T, joblib
from PIL import Image
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
bb  = torch.load(MODELS + 'sex_dinov2_vits14_backbone.pt', weights_only=False).eval().to(DEV)
clf = joblib.load(MODELS + 'sex_logreg.joblib')
tf_ = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
with torch.no_grad():
    ext['p_male'] = [clf.predict_proba(bb(torch.stack([tf_(Image.open(f).convert('RGB')) for f in frames(d, 8)]).to(DEV))
                                       .float().cpu().numpy())[:, 1].mean() for d in ext.path]

ext.drop(columns='path').to_csv(PRED_FILE, index=False)
print('Saved', PRED_FILE)

# ---------- 5. pre-declared metrics ----------
from sklearn.metrics import roc_auc_score
def metrics(t, p):
    yhat = p > .5; pos, neg = t == 1, t == 0
    r1 = yhat[pos].mean() if pos.any() else np.nan; r0 = (~yhat[neg]).mean() if neg.any() else np.nan
    return dict(n=len(t), n_pos=int(pos.sum()), acc=(yhat == t).mean(), bal=np.nanmean([r1, r0]),
                recall_pos=r1, recall_neg=r0, majority=max(pos.mean(), neg.mean()),
                auc=roc_auc_score(t, p) if 0 < pos.sum() < len(t) else np.nan)
def boot_ci(df, tcol, pcol, B=2000):
    rng = np.random.default_rng(0); v = df.video.unique(); g = {k: x for k, x in df.groupby('video')}; out = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(v, len(v))]); t = s[tcol].values
        if 0 < t.sum() < len(t): out.append(metrics(t, s[pcol].values)['bal'])
    return np.percentile(out, [2.5, 97.5])

ext['y_wave'] = (ext.activity == 'waving').astype(int); ext['y_male'] = (ext.sex == 'male').astype(int)
rows = []
for set_name, keep in [('A_all', np.ones(64, bool)), ('B_blindQC', ~ext.blind_exclude.values), ('C_strict', ~ext.strict_exclude.values)]:
    d = ext[keep]
    for task, tcol, pcol, sub in [('activity_aug (primary)', 'y_wave', 'p_waving_aug', None),
                                  ('activity_noaug', 'y_wave', 'p_waving_noaug', None),
                                  ('sex', 'y_male', 'p_male', None),
                                  ('sex_feeding_only', 'y_male', 'p_male', 'feeding')]:
        dd = d if sub is None else d[d.activity == sub]
        m = metrics(dd[tcol].values, dd[pcol].values); lo, hi = boot_ci(dd, tcol, pcol)
        rows.append(dict(set=set_name, task=task, **m, bal_ci_lo=lo, bal_ci_hi=hi))
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT + 'external_metrics_LOCKED.csv', index=False)
pd.set_option('display.width', 200); print(res.to_string(index=False))
print('\nDone. Send me external_predictions_LOCKED.csv, external_metrics_LOCKED.csv and ANALYSIS_PLAN.json from', OUT)
