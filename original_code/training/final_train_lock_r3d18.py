# =====================================================================
# FINAL R3D-18 ACTIVITY MODEL  ->  train on all 278 sequences and LOCK
# (run ONCE, before any v3 field video exists)
#
# Per seed (1 and 2):
#   Step A: train on folds 1-4, early-stop on fold 0  -> best epoch E
#   Step B: retrain from scratch on ALL 278 sequences for exactly E epochs -> save
# Same settings as the grouped-CV baseline run (24 frames, 112 px, Kinetics norm,
# h-flip + brightness, AdamW 1e-4, class weights, two heads: activity + sex).
# The locked prediction = mean p(waving) of the two seeds; waving if > 0.5.
#
# Colab: Runtime > Change runtime type > GPU.  Just run this one cell.
# =====================================================================
import os, json, hashlib, zipfile, random, time, datetime, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/Crab behaviour analysis_UniMAP & NSTRU/Video/Processed_dataset/'
ZIP       = DRIVE_DIR + 'crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
MANIFEST  = DRIVE_DIR + 'sequence_manifest_with_folds.csv'
OUT       = '/content/drive/MyDrive/Crab_Final_October/r3d18_locked/'
LOCKFILE  = '/content/drive/MyDrive/Crab_Final_October/LOCK_r3d18.json'
SEEDS = [1, 2]
SEQ_LEN, MAX_EPOCHS, PATIENCE, LR, BATCH = 24, 40, 8, 1e-4, 4
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

assert DEV == 'cuda', 'Switch the Colab runtime to GPU first.'
assert os.path.exists(ZIP), 'Dataset zip not found: ' + ZIP
assert os.path.exists(MANIFEST), 'Manifest not found: ' + MANIFEST
assert not os.path.exists(LOCKFILE), 'R3D-18 is already locked (' + LOCKFILE + '). Do not overwrite.'
os.makedirs(OUT, exist_ok=True)

# ---------- data ----------
if not os.path.exists('/content/ds'): zipfile.ZipFile(ZIP).extractall('/content/ds')
BASE = '/content/ds/crab_behavior_processed_data_waving_feeding_no_others_QC'
man = pd.read_csv(MANIFEST)
man['path'] = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
missing = [p for p in man.path if not os.path.isdir(p)]
assert not missing, f'{len(missing)} sequence folders missing, e.g. {missing[:3]}'

def load_seq(d):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.lower().endswith(('.jpg', '.png')))
    f = [f[i] for i in np.linspace(0, len(f) - 1, SEQ_LEN).astype(int)]
    return np.stack([np.array(Image.open(p).convert('RGB').resize((224, 224))) for p in f])
print('loading frames ...'); X = np.stack([load_seq(d) for d in man.path]); print(X.shape)
Y_ACT = torch.tensor((man.activity == 'waving').values.astype(np.int64))
Y_SEX = torch.tensor((man.sex == 'male').values.astype(np.int64))
KIN_M, KIN_S = torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989])

def batch_tensor(idx, train):
    x = torch.from_numpy(X[idx]).float().div(255)
    if train:
        for b in range(len(idx)):
            if random.random() < 0.5: x[b] = x[b].flip(2)
            x[b] = (x[b] * random.uniform(0.8, 1.2)).clamp(0, 1)
    x = ((x - KIN_M) / KIN_S).permute(0, 4, 1, 2, 3)
    x = F.interpolate(x, size=(SEQ_LEN, 112, 112), mode='trilinear', align_corners=False)
    return x.to(DEV, non_blocking=True)

class R3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torchvision.models.video.r3d_18(weights='KINETICS400_V1'); self.b.fc = nn.Identity()
        self.drop = nn.Dropout(0.5); self.act = nn.Linear(512, 2); self.sex = nn.Linear(512, 2)
    def forward(self, x):
        h = self.drop(self.b(x)); return self.act(h), self.sex(h)

def bal(t, p):
    y = p > .5; r = [y[t == 1].mean() if (t == 1).any() else np.nan, (~y[t == 0]).mean() if (t == 0).any() else np.nan]
    return float(np.nanmean(r))

@torch.no_grad()
def predict(model, idx):
    model.eval(); pa, ps = [], []
    for i in range(0, len(idx), BATCH):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            a, s = model(batch_tensor(idx[i:i + BATCH], False))
        pa.append(a.float().softmax(1)[:, 1].cpu()); ps.append(s.float().softmax(1)[:, 1].cpu())
    return torch.cat(pa).numpy(), torch.cat(ps).numpy()

def train(tr, seed, epochs=None, va=None):
    """epochs=None -> early stopping on va, returns best epoch. epochs=E -> train exactly E epochs, returns model."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    wa = torch.tensor(len(tr) / (2 * np.bincount(Y_ACT[tr].numpy(), minlength=2)), dtype=torch.float32, device=DEV)
    ws = torch.tensor(len(tr) / (2 * np.bincount(Y_SEX[tr].numpy(), minlength=2)), dtype=torch.float32, device=DEV)
    model = R3D().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=DEV == 'cuda')
    best, best_ep, wait = -1, 1, 0
    for ep in range(epochs or MAX_EPOCHS):
        model.train(); perm = np.random.permutation(tr); tot = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
                a, s = model(batch_tensor(b, True))
                loss = F.cross_entropy(a.float(), Y_ACT[b].to(DEV), weight=wa) + \
                       F.cross_entropy(s.float(), Y_SEX[b].to(DEV), weight=ws)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
        if epochs is None:
            pa, ps = predict(model, va)
            score = (bal(Y_ACT[va].numpy(), pa) + bal(Y_SEX[va].numpy(), ps)) / 2
            print(f'  seed{seed} [find epoch] ep{ep+1:02d} loss {tot/len(tr):.3f} val_bal {score:.3f}')
            if score > best: best, best_ep, wait = score, ep + 1, 0
            else:
                wait += 1
                if wait >= PATIENCE: break
        else:
            print(f'  seed{seed} [final, all data] ep{ep+1:02d}/{epochs} loss {tot/len(tr):.3f}')
    return (best_ep, best) if epochs is None else model

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

ALL = np.arange(len(man)); info = {}
for seed in SEEDS:
    t0 = time.time()
    E, vscore = train(np.where(man.fold != 0)[0], seed, va=np.where(man.fold == 0)[0])
    print(f'seed{seed}: best epoch {E} (val bal {vscore:.3f}) -> retraining on all {len(ALL)} sequences')
    model = train(ALL, seed, epochs=E)
    pa, ps = predict(model, ALL)
    p = OUT + f'activity_r3d18_seed{seed}.pt'
    torch.save(model.state_dict(), p)                      # state_dict only: load with R3D() defined above
    info[f'activity_r3d18_seed{seed}.pt'] = dict(sha256=sha(p), epochs=int(E), val_bal_fold0=round(float(vscore), 4),
                                                 train_bal_activity=round(bal(Y_ACT.numpy(), pa), 4),
                                                 minutes=round((time.time() - t0) / 60, 1))
    print(f'seed{seed} saved  train activity bal {bal(Y_ACT.numpy(), pa):.3f}  ({(time.time()-t0)/60:.1f} min)')
    del model; torch.cuda.empty_cache()

lock = dict(
    locked_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S UTC'),
    purpose='Pre-declared primary ACTIVITY model for external set v3 (EXTERNAL_V2_PLAN.md, Amendment 2)',
    files=info, folder=OUT, n_train_sequences=int(len(man)),
    architecture='torchvision r3d_18 (Kinetics-400 init), fc=Identity, Dropout 0.5, heads act=Linear(512,2), sex=Linear(512,2)',
    preprocessing='24 evenly spaced frames; RGB; resize 224x224; /255; Kinetics mean/std; trilinear resize to 24x112x112',
    decision='mean softmax p(waving) over the 2 seeds; waving if > 0.5',
    sex_head_note='sex head is NOT used; sex model stays DINOv2 + logistic regression (LOCK.json)',
    training=dict(optimizer='AdamW', lr=LR, weight_decay=1e-4, batch=BATCH, fp16=True, class_weighted_ce=True,
                  augmentation='h-flip p=0.5, brightness x U(0.8,1.2), per clip',
                  epoch_rule='early stopping (patience 8) on fold 0 with folds 1-4, then retrain on all data for that many epochs'),
    torch=torch.__version__, torchvision=torchvision.__version__)
with open(LOCKFILE, 'w') as f: json.dump(lock, f, indent=2)
print(json.dumps(lock, indent=2))
print('\nLOCKED. Send me', LOCKFILE)
