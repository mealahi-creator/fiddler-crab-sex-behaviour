# =====================================================================
# EXTERNAL TEST v2 — the re-annotated, re-processed external clips.
# Run ONCE, only AFTER the blind QC list for v2 is locked. No tuning, no re-runs.
#
# Drive needs:
#   MyDrive/Crab_Final_October/                    6 locked models + LOCK.json (unchanged)
#   MyDrive/Crab_Final_October/LOCKED_external_v2_exclusions.csv   (from the v2 blind QC)
#   MyDrive/Crab_External_v2/Processed_dataset/    output of Video_Preprocessing_Script
#        <sex>/<activity>/<clip folder>/*.jpg
#
# Pre-declared (EXTERNAL_V2_PLAN.md; written to ANALYSIS_PLAN_v2.json before any prediction):
#   Clip folders with fewer than 2 frames are excluded (no detections = training cleaning rule).
#   Sets: A = all usable clips | B = clips kept by blind QC
#   Activity: primary aug (mean of 2 seeds), secondary no-aug; waving if p > 0.5
#   Sex: DINOv2 + logistic regression; male if p > 0.5; also on feeding clips only
#   Metrics: accuracy, balanced accuracy, per-class recall, majority baseline, AUC,
#            95% CI by bootstrapping videos (2000 resamples, seed 0)
#   Model input identical to training: 24 evenly spaced frames, resize 224.
# =====================================================================
import os, re, json, glob, hashlib, datetime, numpy as np, pandas as pd, cv2, tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')

MODELS = '/content/drive/MyDrive/Crab_Final_October/'
DATA   = '/content/drive/MyDrive/Crab_External_v2/Processed_dataset/'
QCFILE = MODELS + 'LOCKED_external_v2_exclusions.csv'
OUT    = MODELS + 'external_test_v2/'
os.makedirs(OUT, exist_ok=True)
PRED_FILE = OUT + 'external_v2_predictions_LOCKED.csv'
assert not os.path.exists(PRED_FILE), 'External v2 test already run once. Do not re-run; send me the existing file.'
assert os.path.exists(QCFILE), 'Blind QC for v2 is not locked yet. Do the QC first, then run this.'

# ---------- 1. models are exactly the locked ones ----------
lock = json.load(open(MODELS + 'LOCK.json'))
for f, h in lock['sha256'].items():
    assert hashlib.sha256(open(MODELS + f, 'rb').read()).hexdigest() == h, f'{f} changed since lock!'
print('All 6 model files match LOCK.json')

# ---------- 2. clips, QC, plan ----------
recs = []
for d in sorted(glob.glob(DATA + '*/*/*/')):
    sex, act, clip = d.rstrip('/').split('/')[-3:]
    if sex not in ('male', 'female') or act not in ('waving', 'feeding'): continue
    n = len([x for x in os.listdir(d) if x.endswith(('.jpg', '.png'))])
    m = re.match(r'(MVI|IMG)_\d+', clip)
    recs.append(dict(seq_name=clip, sex=sex, activity=act, video=m.group(0) if m else clip, n_frames=n, path=d))
ext = pd.DataFrame(recs)
print(len(ext), 'clip folders found;', (ext.n_frames < 2).sum(), 'have < 2 frames and are excluded (pre-declared)')
ext[ext.n_frames < 2].drop(columns='path').to_csv(OUT + 'excluded_too_few_frames.csv', index=False)
ext = ext[ext.n_frames >= 2].reset_index(drop=True)

qc = pd.read_csv(QCFILE)
assert {'seq_name', 'blind_exclude'} <= set(qc.columns), 'QC file needs columns seq_name, blind_exclude'
ext = ext.merge(qc[['seq_name', 'blind_exclude']], on='seq_name', how='left')
assert ext.blind_exclude.notna().all(), f'Clips missing from the QC list: {ext[ext.blind_exclude.isna()].seq_name.tolist()}'
ext['blind_exclude'] = ext.blind_exclude.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])

plan = {'written_utc': datetime.datetime.utcnow().isoformat(timespec='seconds'), 'lock_utc': lock['locked_at_utc'],
        'sets': {'A_all': len(ext), 'B_blindQC': int((~ext.blind_exclude).sum())},
        'excluded_B': sorted(ext.seq_name[ext.blind_exclude]),
        'class_counts_A': {f'{a}_{b}': int(v) for (a, b), v in ext.groupby(['sex', 'activity']).size().items()},
        'n_videos': int(ext.video.nunique()), 'thresholds': 0.5,
        'preprocessing': 'Video_Preprocessing_Script unchanged; model input 24 evenly spaced frames, resize 224',
        'metrics': 'acc, balanced acc, per-class recall, majority baseline, AUC, video-bootstrap 95% CI (2000, seed 0)'}
json.dump(plan, open(OUT + 'ANALYSIS_PLAN_v2.json', 'w'), indent=2)
print('Plan written:', plan['sets'], plan['class_counts_A'])

# ---------- 3. activity ----------
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

# ---------- 4. sex ----------
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
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

ext['y_wave'] = (ext.activity == 'waving').astype(int); ext['y_male'] = (ext.sex == 'male').astype(int)
rows = []
for set_name, keep in [('A_all', np.ones(len(ext), bool)), ('B_blindQC', ~ext.blind_exclude.values)]:
    d = ext[keep]
    for task, tcol, pcol, sub in [('activity_aug (primary)', 'y_wave', 'p_waving_aug', None),
                                  ('activity_noaug', 'y_wave', 'p_waving_noaug', None),
                                  ('sex', 'y_male', 'p_male', None),
                                  ('sex_feeding_only', 'y_male', 'p_male', 'feeding')]:
        dd = d if sub is None else d[d.activity == sub]
        m = metrics(dd[tcol].values, dd[pcol].values); lo, hi = boot_ci(dd, tcol, pcol)
        rows.append(dict(set=set_name, task=task, **m, bal_ci_lo=lo, bal_ci_hi=hi))
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT + 'external_v2_metrics_LOCKED.csv', index=False)
pd.set_option('display.width', 200); print(res.to_string(index=False))
print('\nDone. Send me everything in', OUT)
