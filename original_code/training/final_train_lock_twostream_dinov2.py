# =====================================================================
# FINAL MODELS — train on all 278 training sequences, save, LOCK.
# Nothing in this script reads the external set.
#
# Colab, GPU runtime. Upload to /content/:
#   crab_behavior_processed_data_waving_feeding_no_others_QC.zip
#   sequence_manifest_with_folds.csv
# Output: /content/drive/MyDrive/Crab_Final/   (models + LOCK.json)
# Re-running skips anything already saved.
#
# Pre-declared rules (write these in Methods exactly as below):
#  * Activity: same two-stream model and hyper-parameters as the grouped CV.
#    No validation set is held out, so training runs for a FIXED number of
#    epochs = median best epoch across the 10 grouped-CV runs
#    (augmented: 34, no-aug: 32). Constant LR 1.25e-4, no early stopping.
#    Two seeds (1, 2); the final p_waving is the mean of the two seeds.
#    Primary = augmented. Secondary (declared now) = no-aug.
#    The sex head of this network is NOT used.
#  * Sex: frozen DINOv2 ViT-S/14, full crop resized to 224, 8 evenly spaced
#    frames; StandardScaler + LogisticRegression(C=0.1, balanced), trained on
#    all feeding sequences; p_male = mean over 8 frames; male if > 0.5.
# =====================================================================
import os, json, zipfile, random, hashlib, datetime, numpy as np, pandas as pd, cv2, tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import CSVLogger
from google.colab import drive
drive.mount('/content/drive')

ZIP      = '/content/crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
MANIFEST = '/content/sequence_manifest_with_folds.csv'
OUT      = '/content/drive/MyDrive/Crab_Final/'
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 2]
EPOCHS = {'aug': 34, 'noaug': 32}          # median best epoch from grouped CV logs
LR, SEQ_LEN, DIM, BATCH = 1.25e-4, 24, (224, 224), 4

if not os.path.exists('/content/ds'):
    zipfile.ZipFile(ZIP).extractall('/content/ds')
BASE = '/content/ds/crab_behavior_processed_data_waving_feeding_no_others_QC'
man = pd.read_csv(MANIFEST)
man['path'] = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
man['y_act'] = (man.activity == 'waving').astype(int)
man['y_sex'] = (man.sex == 'male').astype(int)
assert len(man) == 278, len(man)

# ---------------- PART 1: activity models (two-stream) ----------------
def frame_paths(d):
    p = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.jpg', '.png')))
    return [p[i] for i in np.linspace(0, len(p) - 1, SEQ_LEN, dtype=int)]

def make_inputs(paths, train, res_aug):
    flip = train and random.random() < 0.5
    gain = random.uniform(0.8, 1.2) if train else 1.0
    deg  = random.choice([None, 16, 24, 32, 48, 64, 96]) if (train and res_aug) else None
    rgb, dif, prev = [], [], None
    for p in paths:
        f = cv2.resize(cv2.imread(p), DIM)
        if flip: f = cv2.flip(f, 1)
        if gain != 1.0: f = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if deg: f = cv2.resize(cv2.resize(f, (deg, deg), interpolation=cv2.INTER_AREA), DIM, interpolation=cv2.INTER_LINEAR)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        rgb.append(f.astype(np.float32) / 255.0)
        if prev is None: prev = g
        dif.append((cv2.absdiff(g, prev).astype(np.float32) / 255.0)[..., None]); prev = g
    return np.array(rgb), np.array(dif)

class Gen(tf.keras.utils.Sequence):
    def __init__(self, df, train, res_aug, weights=None):
        super().__init__(); self.df = df.reset_index(drop=True); self.train = train
        self.res_aug = res_aug; self.w = weights
        self.paths = [frame_paths(p) for p in self.df.path]
        self.idx = np.arange(len(self.df)); self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.df) / BATCH))
    def on_epoch_end(self):
        if self.train: np.random.shuffle(self.idx)
    def __getitem__(self, i):
        b = self.idx[i * BATCH:(i + 1) * BATCH]
        R, D = zip(*[make_inputs(self.paths[j], self.train, self.res_aug) for j in b])
        x = {'rgb_input': np.array(R), 'diff_input': np.array(D)}
        y = {'activity_output': tf.keras.utils.to_categorical(self.df.y_act.values[b], 2),
             'gender_output': self.df.y_sex.values[b].reshape(-1, 1).astype(np.float32)}
        if self.train and self.w is not None:
            return x, y, np.array([self.w[k] for k in (self.df.sex + '_' + self.df.activity).values[b]])
        return x, y

def stream(x, name):
    def cbr(x, f, k, s=(1, 1, 1)):
        x = layers.Conv3D(f, k, strides=s, padding='same', use_bias=False)(x)
        return layers.ReLU()(layers.BatchNormalization()(x))
    def block(x, f, s=(1, 1, 1)):
        idt = x; o = cbr(x, f, (3, 3, 3), s)
        o = layers.BatchNormalization()(layers.Conv3D(f, (3, 3, 3), padding='same', use_bias=False)(o))
        if s != (1, 1, 1) or x.shape[-1] != f:
            idt = layers.BatchNormalization()(layers.Conv3D(f, 1, strides=s, use_bias=False)(x))
        return layers.ReLU()(layers.Add()([o, idt]))
    x = cbr(x, 32, (3, 7, 7), (1, 2, 2)); x = layers.MaxPooling3D((1, 3, 3), (1, 2, 2), padding='same')(x)
    x = block(x, 32); x = block(x, 64, (2, 2, 2)); x = block(x, 128, (1, 2, 2))
    return layers.GlobalAveragePooling3D(name=f'{name}_gap')(x)

def build():
    r = layers.Input((SEQ_LEN, *DIM, 3), name='rgb_input'); d = layers.Input((SEQ_LEN, *DIM, 1), name='diff_input')
    reg = regularizers.l2(0.005)
    s = layers.Dropout(0.5)(layers.Dense(128, 'relu', kernel_regularizer=reg)(stream(r, 'spatial')))
    t = layers.Dropout(0.5)(layers.Dense(128, 'relu', kernel_regularizer=reg)(stream(d, 'temporal')))
    m = Model([r, d], [layers.Dense(2, 'softmax', name='activity_output')(t),
                       layers.Dense(1, 'sigmoid', name='gender_output')(s)])
    m.compile(tf.keras.optimizers.Adam(LR),
              loss={'activity_output': 'categorical_crossentropy', 'gender_output': 'binary_crossentropy'},
              metrics={'activity_output': ['accuracy'], 'gender_output': ['binary_accuracy']})
    return m

comp = man.sex + '_' + man.activity
W = (len(comp) / (comp.nunique() * comp.value_counts())).to_dict()

for variant, res_aug in [('aug', True), ('noaug', False)]:
    for seed in SEEDS:
        tag = f'activity_{variant}_seed{seed}'; path = OUT + tag + '.keras'
        if os.path.exists(path): print('skip', tag); continue
        random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
        model = build()
        model.fit(Gen(man, True, res_aug, W), epochs=EPOCHS[variant], verbose=2,
                  callbacks=[CSVLogger(OUT + f'log_{tag}.csv')])
        model.save(path); print('saved', tag)
        tf.keras.backend.clear_session()

# Sanity check only (training data, so this is NOT a result): should be high.
for variant in ['aug', 'noaug']:
    p = []
    for seed in SEEDS:
        m = tf.keras.models.load_model(OUT + f'activity_{variant}_seed{seed}.keras')
        pr = m.predict(Gen(man, False, False), verbose=0)
        p.append((pr['activity_output'] if isinstance(pr, dict) else pr[0])[:, 1])
        tf.keras.backend.clear_session()
    print(f'[sanity, train data] activity {variant}: acc {((np.mean(p, 0) > .5) == man.y_act).mean():.3f}')

# ---------------- PART 2: sex model (DINOv2 + logistic regression) ----------------
import torch, torchvision.transforms as T, joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEX_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
if not os.path.exists(OUT + 'sex_dinov2_vits14_backbone.pt'):
    bb = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
    torch.save(bb, OUT + 'sex_dinov2_vits14_backbone.pt')       # freeze the exact backbone weights
bb = torch.load(OUT + 'sex_dinov2_vits14_backbone.pt', weights_only=False).eval().to(DEV)

def frames8(d):
    f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
    return [f[i] for i in np.linspace(0, len(f) - 1, 8).astype(int)]

@torch.no_grad()
def feats(dirs):
    return np.stack([bb(torch.stack([SEX_TF(Image.open(x).convert('RGB')) for x in frames8(d)]).to(DEV)).float().cpu().numpy()
                     for d in dirs])                                    # (n, 8, 384)

feed = man[man.activity == 'feeding'].reset_index(drop=True)
F = feats(feed.path); n, k, d = F.shape
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced'))
clf.fit(F.reshape(-1, d), np.repeat(feed.y_sex.values, k))
joblib.dump(clf, OUT + 'sex_logreg.joblib')
print(f'sex classifier trained on {n} feeding sequences '
      f'({(feed.sex=="male").sum()} male, {(feed.sex=="female").sum()} female)')

# Sanity check on all 278 training sequences (NOT a result)
Fa = feats(man.path)
p = clf.predict_proba(Fa.reshape(-1, d))[:, 1].reshape(-1, 8).mean(1)
print(f'[sanity, train data] sex acc {((p > .5) == man.y_sex).mean():.3f}')

# ---------------- PART 3: LOCK ----------------
def sha(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
files = sorted(f for f in os.listdir(OUT) if f.endswith(('.keras', '.joblib', '.pt')))
lock = {
    'locked_at_utc': datetime.datetime.utcnow().isoformat(timespec='seconds'),
    'n_training_sequences': int(len(man)),
    'activity': {'primary': 'aug', 'secondary': 'noaug', 'seeds': SEEDS, 'epochs': EPOCHS,
                 'decision': 'mean p_waving over seeds; waving if > 0.5', 'seq_len': SEQ_LEN, 'size': list(DIM)},
    'sex': {'backbone': 'dinov2_vits14 (frozen)', 'crop': 'full, resize 224', 'frames': 8,
            'classifier': 'StandardScaler + LogisticRegression(C=0.1, balanced)', 'trained_on': 'feeding only',
            'n_train': int(n), 'decision': 'mean p_male over 8 frames; male if > 0.5'},
    'sha256': {f: sha(OUT + f) for f in files},
}
json.dump(lock, open(OUT + 'LOCK.json', 'w'), indent=2)
print(json.dumps(lock, indent=2))
print('\nLOCKED. Send me LOCK.json (and the two log_activity_*_seed*.csv files). Do NOT run anything on the external set yet.')
