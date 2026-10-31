"""Re-fit the image-encoder sex classifiers from the saved frame features (Table S5, DINOv2 column of Table 6).

The frozen DINOv2 ViT-S/14 and ConvNeXt-T features of 8 evenly spaced frames per sequence are stored in
data/features/image_encoder_features.npz (float16). A logistic regression (C = 0.1, balanced class weights,
standardised features) is trained frame-wise on the training folds and the sequence probability of 'male' is the mean over
the 8 frames, exactly as in original_code/training/sex_probe_colab.py. This script repeats that step on a CPU in a
few minutes and compares the result with the saved predictions in data/predictions/cv_sex_image_encoders.csv.
The features are stored at half precision; with them the re-fit reproduces every saved decision (checked in the output column same_decision_pct).
Output: results/tables/TableS5_refit_from_features.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import DATA, TABLES, balanced_accuracy, load_sex_variants, pct

z = np.load(DATA / 'features' / 'image_encoder_features.npz', allow_pickle=True)
saved = load_sex_variants()
man = saved[(saved.backbone == 'dinov2_s') & (saved.crop == 'full') & (saved.train_on == 'feeding_only')].set_index('seq').loc[z['seq']].reset_index()
y = (man.sex == 'male').astype(int).values
feeding = (man.activity == 'feeding').values
rows = []
for bb in ['dinov2_s', 'convnext_t']:
    for crop in ['full', 'tight']:
        F = z[f'{bb}__{crop}'].astype(np.float32)
        n, k, d = F.shape
        for train_on in ['feeding_only', 'all']:
            p = np.zeros(n)
            for fold in range(5):
                tr = (man.fold != fold).values & (feeding if train_on == 'feeding_only' else True)
                te = (man.fold == fold).values
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced'))
                clf.fit(F[tr].reshape(-1, d), np.repeat(y[tr], k))
                p[te] = clf.predict_proba(F[te].reshape(-1, d))[:, 1].reshape(-1, k).mean(1)
            ref = saved[(saved.backbone == bb) & (saved.crop == crop) & (saved.train_on == train_on)].set_index('seq').loc[man.seq, 'p_male_seq'].values
            rows.append(dict(encoder=bb, crop=crop, trained_on=train_on,
                             ba_all_refit=balanced_accuracy(y, p), ba_all_saved=balanced_accuracy(y, ref),
                             ba_feeding_refit=balanced_accuracy(y[feeding], p[feeding]), ba_feeding_saved=balanced_accuracy(y[feeding], ref[feeding]),
                             same_decision_pct=100 * np.mean((p > 0.5) == (ref > 0.5)), max_abs_prob_diff=np.abs(p - ref).max()))
            r = rows[-1]
            print(f"{bb:10s} {crop:5s} {train_on:12s} BA all {pct(r['ba_all_refit'])} (saved {pct(r['ba_all_saved'])})  "
                  f"feeding {pct(r['ba_feeding_refit'])} (saved {pct(r['ba_feeding_saved'])})  same decision {r['same_decision_pct']:.1f}%")
pd.DataFrame(rows).to_csv(TABLES / 'TableS5_refit_from_features.csv', index=False)
