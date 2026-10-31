# Re-extract and SAVE DINOv2 frame features (only needed if the earlier Colab session closed).
# CPU runtime is fine (~5 min). Needs the same two files in /content/ as before.
import os, zipfile, numpy as np, pandas as pd, torch, torchvision.transforms as T
from PIL import Image
from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/Crab_SexProbe/'; os.makedirs(OUT, exist_ok=True)
ZIP = '/content/crab_behavior_processed_data_waving_feeding_no_others_QC.zip'
man = pd.read_csv('/content/sequence_manifest_with_folds.csv')
if not os.path.exists('/content/ds'): zipfile.ZipFile(ZIP).extractall('/content/ds')
BASE = '/content/ds/crab_behavior_processed_data_waving_feeding_no_others_QC'
paths = BASE + '/' + man.sex + '/' + man.activity + '/' + man.seq
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
F = []
with torch.no_grad():
    for i, d in enumerate(paths):
        f = sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(('.jpg', '.png')))
        f = [f[j] for j in np.linspace(0, len(f) - 1, 8).astype(int)]
        F.append(m(torch.stack([tf(Image.open(x).convert('RGB')) for x in f])).numpy())
        if (i + 1) % 50 == 0: print(i + 1)
np.savez_compressed(OUT + 'sex_probe_features.npz', seq=man.seq.values, dinov2_s__full=np.stack(F).astype(np.float16))
print('saved', OUT + 'sex_probe_features.npz')
