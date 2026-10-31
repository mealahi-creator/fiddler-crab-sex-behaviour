# =====================================================================
# BASELINES under the SAME video-grouped 5-fold CV as the proposed model
#   1) R3D-18, Kinetics-400 pretrained (torchvision)        - pretrained VIDEO model
#   2) EfficientNet-B0 (ImageNet) + BiLSTM                  - pretrained FRAME model + temporal RNN
# Same folds, same 24 evenly spaced frames, same augmentation (h-flip, brightness +/-20%),
# validation fold = (k+1) % 5 for early stopping (as for the proposed model), 2 seeds.
#
# Colab: Runtime > Change runtime type > GPU (T4 is fine). Upload to /content/:
#   crab_behavior_processed_data_waving_feeding_no_others_QC.zip
#   sequence_manifest_with_folds.csv
# Results go to MyDrive/Crab_Baselines/ after every fold; re-running skips finished runs.
# =====================================================================
import os, zipfile, random, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

ZIP      = '/content/crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
MANIFEST = '/content/sequence_manifest_with_folds.csv'
OUT      = '/content/drive/MyDrive/Crab_Baselines/'
MODELS   = ['r3d18', 'effb0_bilstm']
SEEDS    = [1, 2]
SEQ_LEN, MAX_EPOCHS, PATIENCE, LR, BATCH = 24, 40, 8, 1e-4, 4
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT, exist_ok=True)
assert DEV == 'cuda', 'Switch the Colab runtime to GPU first.'

# ---------- data: load every sequence once (24 frames, 224 px, uint8) ----------
if not os.path.exists('/content/ds'): zipfile.ZipFile(ZIP).extractall('/content/ds')
BASE = '/content/ds/crab_behavior_processed_data_waving_feeding_no_others_QC'
man = pd.read_csv(MANIFEST)
man['path'] = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
def load_seq(d):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
    f = [f[i] for i in np.linspace(0, len(f) - 1, SEQ_LEN).astype(int)]
    return np.stack([np.array(Image.open(p).convert('RGB').resize((224, 224))) for p in f])   # T,H,W,3
print('loading frames ...'); X = np.stack([load_seq(d) for d in man.path]); print(X.shape)
Y_ACT = torch.tensor((man.activity == 'waving').values.astype(np.int64))
Y_SEX = torch.tensor((man.sex == 'male').values.astype(np.int64))

KIN_M, KIN_S = torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989])
IMN_M, IMN_S = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

def batch_tensor(idx, train, model_name):
    x = torch.from_numpy(X[idx]).float().div(255)                        # B,T,H,W,3
    if train:
        for b in range(len(idx)):
            if random.random() < 0.5: x[b] = x[b].flip(2)                # same flip for whole clip
            x[b] = (x[b] * random.uniform(0.8, 1.2)).clamp(0, 1)        # same brightness for whole clip
    if model_name == 'r3d18':
        x = (x - KIN_M) / KIN_S
        x = x.permute(0, 4, 1, 2, 3)                                    # B,3,T,H,W
        x = F.interpolate(x, size=(SEQ_LEN, 112, 112), mode='trilinear', align_corners=False)   # R3D native size
    else:
        x = ((x - IMN_M) / IMN_S).permute(0, 1, 4, 2, 3)                 # B,T,3,H,W
    return x.to(DEV, non_blocking=True)

# ---------- models: shared backbone, two heads (activity, sex) ----------
class R3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torchvision.models.video.r3d_18(weights='KINETICS400_V1'); self.b.fc = nn.Identity()
        self.drop = nn.Dropout(0.5); self.act = nn.Linear(512, 2); self.sex = nn.Linear(512, 2)
    def forward(self, x):
        h = self.drop(self.b(x)); return self.act(h), self.sex(h)

class EffLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        e = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1'); e.classifier = nn.Identity(); self.e = e
        self.lstm = nn.LSTM(1280, 256, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5); self.act = nn.Linear(512, 2); self.sex = nn.Linear(512, 2)
    def forward(self, x):
        B, T = x.shape[:2]
        f = self.e(x.flatten(0, 1)).view(B, T, -1)
        h, _ = self.lstm(f); h = self.drop(h.mean(1)); return self.act(h), self.sex(h)

def bal(t, p):
    y = p > .5; return 0.5 * (y[t == 1].mean() + (~y[t == 0]).mean())

@torch.no_grad()
def predict(model, idx, name):
    model.eval(); pa, ps = [], []
    for i in range(0, len(idx), BATCH):
        with torch.autocast('cuda', dtype=torch.float16):
            a, s = model(batch_tensor(idx[i:i + BATCH], False, name))
        pa.append(a.float().softmax(1)[:, 1].cpu()); ps.append(s.float().softmax(1)[:, 1].cpu())
    return torch.cat(pa).numpy(), torch.cat(ps).numpy()

# ---------- grouped CV ----------
for name in MODELS:
    for seed in SEEDS:
        for k in range(5):
            tag = f'{name}_fold{k}_seed{seed}'; pf = OUT + f'pred_{tag}.csv'
            if os.path.exists(pf): print('skip', tag); continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            vk = (k + 1) % 5
            tr = np.where(~man.fold.isin([k, vk]))[0]; va = np.where(man.fold == vk)[0]; te = np.where(man.fold == k)[0]
            wa = torch.tensor(len(tr) / (2 * np.bincount(Y_ACT[tr].numpy(), minlength=2)), dtype=torch.float32, device=DEV)
            ws = torch.tensor(len(tr) / (2 * np.bincount(Y_SEX[tr].numpy(), minlength=2)), dtype=torch.float32, device=DEV)
            model = (R3D() if name == 'r3d18' else EffLSTM()).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler()
            best, best_state, wait, t0 = -1, None, 0, time.time()
            for ep in range(MAX_EPOCHS):
                model.train(); perm = np.random.permutation(tr); tot = 0
                for i in range(0, len(perm), BATCH):
                    b = perm[i:i + BATCH]
                    with torch.autocast('cuda', dtype=torch.float16):
                        a, s = model(batch_tensor(b, True, name))
                        loss = F.cross_entropy(a.float(), Y_ACT[b].to(DEV), weight=wa) + \
                               F.cross_entropy(s.float(), Y_SEX[b].to(DEV), weight=ws)
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
                pa, ps = predict(model, va, name)
                score = (bal(Y_ACT[va].numpy(), pa) + bal(Y_SEX[va].numpy(), ps)) / 2       # same criterion as proposed model
                print(f'{tag} ep{ep:02d} loss {tot/len(tr):.3f} val_bal {score:.3f}')
                if score > best: best, wait = score, 0; best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PATIENCE: break
            model.load_state_dict(best_state)
            pa, ps = predict(model, te, name)
            res = man.iloc[te][['seq', 'video', 'sex', 'activity', 'fold']].copy()
            res['p_waving'], res['p_male'], res['seed'], res['model'] = pa, ps, seed, name
            res.to_csv(pf, index=False)
            print(f'=== {tag}: activity bal {bal(Y_ACT[te].numpy(), pa):.3f}  sex bal {bal(Y_SEX[te].numpy(), ps):.3f}  '
                  f'({(time.time()-t0)/60:.1f} min)')
            del model, opt; torch.cuda.empty_cache()

# ---------- pooled summary (seed-averaged, same format as the figure data) ----------
from sklearn.metrics import roc_auc_score
rows = []
for name in MODELS:
    fs = [OUT + f'pred_{name}_fold{k}_seed{s}.csv' for k in range(5) for s in SEEDS]
    if not all(os.path.exists(f) for f in fs): print(name, 'not finished'); continue
    d = pd.concat(pd.read_csv(f) for f in fs).groupby(['seq', 'video', 'sex', 'activity', 'fold'], as_index=False)[['p_waving', 'p_male']].mean()
    d.to_csv(OUT + f'cv_{name}.csv', index=False)
    ta, ts = (d.activity == 'waving').values.astype(int), (d.sex == 'male').values.astype(int); f = d.activity == 'feeding'
    fold_a = [bal(ta[d.fold == k], d.p_waving.values[d.fold == k]) for k in range(5)]
    fold_s = [bal(ts[d.fold == k], d.p_male.values[d.fold == k]) for k in range(5)]
    rows.append(dict(model=name, activity_bal=bal(ta, d.p_waving.values), activity_sd=np.std(fold_a, ddof=1),
                     activity_auc=roc_auc_score(ta, d.p_waving), sex_bal=bal(ts, d.p_male.values), sex_sd=np.std(fold_s, ddof=1),
                     sex_auc=roc_auc_score(ts, d.p_male), sex_feeding_bal=bal(ts[f], d.p_male.values[f])))
summ = pd.DataFrame(rows).round(3); summ.to_csv(OUT + 'baselines_summary.csv', index=False)
print(summ.to_string(index=False))
print('\nReference (proposed): activity bal 0.777 (aug) / 0.811 (no-aug); sex two-stream 0.704; sex DINOv2 0.948')
print('Send me cv_r3d18.csv, cv_effb0_bilstm.csv and baselines_summary.csv from', OUT)
