# =====================================================================
# SEX IDENTIFICATION — pretrained per-frame features + per-crab aggregation
# Video-grouped 5-fold CV (same folds as the two-stream model).
#
# Run in a SEPARATE Colab notebook on a CPU runtime (or GPU if free).
# Upload to /content/:
#   crab_behavior_processed_data_waving_feeding_no_others_QC.zip
#   sequence_manifest_with_folds.csv
# Output: /content/drive/MyDrive/Crab_SexProbe/sex_probe_predictions.csv  -> send it back
# Pre-declared variants: 2 backbones x 2 crops x 2 training sets (8 total). Do not add more.
# =====================================================================
import os, zipfile, numpy as np, pandas as pd, torch, torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
try:
    from google.colab import drive; drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/Crab_SexProbe/'
except Exception:
    OUT = './Crab_SexProbe/'
os.makedirs(OUT, exist_ok=True)

ZIP      = '/content/crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
MANIFEST = '/content/sequence_manifest_with_folds.csv'
EXTRACT  = '/content/ds'
N_FRAMES = 8
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)

if not os.path.exists(EXTRACT):
    zipfile.ZipFile(ZIP).extractall(EXTRACT)
BASE = os.path.join(EXTRACT, 'crab_behavior_processed_data_waving_feeding_no_others_QC')
man = pd.read_csv(MANIFEST)
man['path'] = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
y = (man.sex == 'male').astype(int).values

def frame_files(d):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
    return [f[i] for i in np.linspace(0, len(f) - 1, N_FRAMES).astype(int)]
FRAMES = [frame_files(p) for p in man.path]

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CROPS = {
    'full':  T.Compose([T.Resize((224, 224)), T.ToTensor(), norm]),
    'tight': T.Compose([T.Resize((320, 320)), T.CenterCrop(224), T.ToTensor(), norm]),   # central 70%
}

def load_backbone(name):
    if name == 'dinov2_s':
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')          # CLS token, 384-d
    elif name == 'convnext_t':
        m = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')            # 768-d pooled
        m.classifier[2] = torch.nn.Identity()
    return m.eval().to(DEV)

@torch.no_grad()
def extract(model, tf):
    feats = []
    for i, files in enumerate(FRAMES):
        x = torch.stack([tf(Image.open(f).convert('RGB')) for f in files]).to(DEV)
        feats.append(model(x).float().cpu().numpy())
        if (i + 1) % 50 == 0: print(f'  {i+1}/{len(FRAMES)}')
    return np.stack(feats)            # (n_seq, N_FRAMES, D)

FEATS = {}
for bb in ['dinov2_s', 'convnext_t']:
    model = load_backbone(bb)
    for cr, tf in CROPS.items():
        print('extracting', bb, cr); FEATS[(bb, cr)] = extract(model, tf)
    del model

rows = []
for (bb, cr), F in FEATS.items():
    n, k, d = F.shape
    for train_on in ['all', 'feeding_only']:
        p_seq = np.zeros(n)
        for fold in range(5):
            tr = (man.fold != fold).values
            if train_on == 'feeding_only': tr &= (man.activity == 'feeding').values
            te = (man.fold == fold).values
            Xtr = F[tr].reshape(-1, d); ytr = np.repeat(y[tr], k)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced'))
            clf.fit(Xtr, ytr)
            p_seq[te] = clf.predict_proba(F[te].reshape(-1, d))[:, 1].reshape(-1, k).mean(1)
        out = man[['seq', 'video', 'crab_track', 'sex', 'activity', 'fold']].copy()
        out['p_male_seq'] = p_seq
        out['p_male_crab'] = out.groupby('crab_track').p_male_seq.transform('mean')   # crab = same video + tracker ID (same fold by construction)
        out['backbone'], out['crop'], out['train_on'] = bb, cr, train_on
        rows.append(out)

res = pd.concat(rows)
res.to_csv(OUT + 'sex_probe_predictions.csv', index=False)

def bal(t, p): return 0.5 * (((p > .5) & (t == 1)).sum() / (t == 1).sum() + ((p <= .5) & (t == 0)).sum() / (t == 0).sum())
summ = []
for key, g in res.groupby(['backbone', 'crop', 'train_on']):
    t = (g.sex == 'male').astype(int).values; f = (g.activity == 'feeding').values
    for lvl in ['p_male_seq', 'p_male_crab']:
        p = g[lvl].values
        summ.append(dict(backbone=key[0], crop=key[1], train_on=key[2], level=lvl.replace('p_male_', ''),
                         bal_all=bal(t, p), auc_all=roc_auc_score(t, p), bal_feeding=bal(t[f], p[f]), auc_feeding=roc_auc_score(t[f], p[f])))
summ = pd.DataFrame(summ).round(3)
summ.to_csv(OUT + 'sex_probe_summary.csv', index=False)
print(summ.to_string(index=False))
print('\nReference (two-stream, grouped CV): bal_all 0.704, auc_all 0.79, bal_feeding 0.620, auc_feeding 0.69')
print('Send', OUT + 'sex_probe_predictions.csv')
