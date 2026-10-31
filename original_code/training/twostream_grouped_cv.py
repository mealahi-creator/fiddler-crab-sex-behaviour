# =====================================================================
# GROUPED 5-FOLD CROSS-VALIDATION (split by source video) — Colab cell
# Same two-stream model as the manuscript, plus resolution augmentation.
#
# Upload to /content/:
#   crab_behavior_processed_data_waving_feeding_no_others_QC.zip
#   sequence_manifest_with_folds.csv
# Everything (predictions, logs, models) is saved to Drive after each run,
# so if Colab disconnects, re-run the cell and finished runs are skipped.
# =====================================================================
import os, zipfile, random, json, numpy as np, pandas as pd, cv2, tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from google.colab import drive
drive.mount('/content/drive')

ZIP      = '/content/crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
MANIFEST = '/content/sequence_manifest_with_folds.csv'
OUT      = '/content/drive/MyDrive/Crab_GroupedCV/'
os.makedirs(OUT, exist_ok=True)

SEEDS        = [1, 2]        # add 3 if time allows
MAX_EPOCHS   = 80
PATIENCE     = 20
LR           = 1.25e-4
SEQ_LEN, DIM = 24, (224, 224)
BATCH        = 4
RES_AUG      = True          # random down/up-scaling to make the model robust to crop resolution

if not os.path.exists('/content/ds'):
    zipfile.ZipFile(ZIP).extractall('/content/ds')
BASE = '/content/ds/crab_behavior_processed_data_waving_feeding_no_others_QC'
man = pd.read_csv(MANIFEST)
man['path'] = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
man['y_act'] = (man.activity == 'waving').astype(int)
man['y_sex'] = (man.sex == 'male').astype(int)

def frame_paths(d):
    p = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.jpg', '.png')))
    return [p[i] for i in np.linspace(0, len(p) - 1, SEQ_LEN, dtype=int)]

def make_inputs(paths, train):
    flip = train and random.random() < 0.5
    gain = random.uniform(0.8, 1.2) if train else 1.0
    deg  = random.choice([None, 16, 24, 32, 48, 64, 96]) if (train and RES_AUG) else None   # same for whole clip
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
    def __init__(self, df, train, weights=None):
        super().__init__(); self.df = df.reset_index(drop=True); self.train = train; self.w = weights
        self.paths = [frame_paths(p) for p in self.df.path]
        self.idx = np.arange(len(self.df)); self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.df) / BATCH))
    def on_epoch_end(self):
        if self.train: np.random.shuffle(self.idx)
    def __getitem__(self, i):
        b = self.idx[i * BATCH:(i + 1) * BATCH]; R, D = zip(*[make_inputs(self.paths[j], self.train) for j in b])
        x = {'rgb_input': np.array(R), 'diff_input': np.array(D)}
        y = {'activity_output': tf.keras.utils.to_categorical(self.df.y_act.values[b], 2),
             'gender_output': self.df.y_sex.values[b].reshape(-1, 1).astype(np.float32)}
        if self.train and self.w is not None:
            sw = np.array([self.w[k] for k in (self.df.sex + '_' + self.df.activity).values[b]])
            return x, y, sw
        return x, y

def stream(x, name):                       # identical to the manuscript model
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

class Combined(Callback):
    def on_epoch_end(self, e, logs=None):
        logs['val_combined_accuracy'] = (logs['val_activity_output_accuracy'] + logs['val_gender_output_binary_accuracy']) / 2

for seed in SEEDS:
    for k in range(5):
        tag = f'fold{k}_seed{seed}'; pred_file = OUT + f'pred_{tag}.csv'
        if os.path.exists(pred_file): print('skip', tag); continue
        random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
        vk = (k + 1) % 5                                   # validation = next fold (also video-disjoint)
        tr, va, te = man[~man.fold.isin([k, vk])], man[man.fold == vk], man[man.fold == k]
        comp = tr.sex + '_' + tr.activity
        w = (len(comp) / (comp.nunique() * comp.value_counts())).to_dict()
        model = build()
        model.fit(Gen(tr, True, w), validation_data=Gen(va, False), epochs=MAX_EPOCHS, verbose=2,
                  callbacks=[Combined(),
                             EarlyStopping('val_combined_accuracy', mode='max', patience=PATIENCE, restore_best_weights=True),
                             ReduceLROnPlateau('val_loss', factor=0.5, patience=10),
                             CSVLogger(OUT + f'log_{tag}.csv')])
        pr = model.predict(Gen(te, False), verbose=0)
        pa, ps = (pr['activity_output'], pr['gender_output']) if isinstance(pr, dict) else (pr[0], pr[1])
        res = te[['seq', 'video', 'sex', 'activity', 'fold']].copy()
        res['p_waving'] = pa[:, 1]; res['p_male'] = ps[:, 0]; res['seed'] = seed
        res.to_csv(pred_file, index=False)
        model.save(OUT + f'model_{tag}.keras')
        acc_a = ((res.p_waving > .5) == (res.activity == 'waving')).mean(); acc_s = ((res.p_male > .5) == (res.sex == 'male')).mean()
        print(f'=== {tag}: activity {acc_a:.3f}  sex {acc_s:.3f}  (n={len(res)})')
        tf.keras.backend.clear_session()

print('All runs done. Send me the pred_*.csv and log_*.csv files from', OUT)
