"""Figures 3-8 and Supplementary Figures S1-S4.

Each figure is saved as PNG (600 dpi), PDF and SVG (editable vector), with <name>_data.csv holding the plotted values.
Figures 1 (map), 2 (example frames) and S5 (original YOLOv11 training curves) are images, provided in figures_static/.
Output: results/figures/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from common import *

OUT = FIGURES
COL = {R3D: '#2a78d6', EFF: '#eb6834', TSA: '#1baf7a', TS: '#eda100', DINO: '#e87ba4'}
MARK = {R3D: 'o', EFF: 's', TSA: 'D', TS: 'v', DINO: '^'}
LS = {R3D: '-', EFF: (0, (5, 2)), TSA: (0, (1, 1.5)), TS: (0, (6, 2, 1, 2)), DINO: '-'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'legend.frameon': False, 'savefig.bbox': 'tight',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95'])


def save(fig, name, table=None):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600 if ext == 'png' else None)
    if table is not None:
        table.to_csv(OUT / f'{name}_data.csv', index=False)
    plt.close(fig)
    print('saved', name)


cv = load_cv()
feed = (cv.activity == 'feeding').values
e71, e90, e64 = load_external('primary71'), load_external('secondary90'), load_external('mismatched64')

# ================================================================ Figure 3 - workflow
def box(ax, x, y, w, h, title, body, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015',
                                fc='#eef4fc' if accent else '#f6f6f4', ec='#2a78d6' if accent else '#b8b7b1', lw=1.2))
    ax.text(x + 0.015, y + h - 0.045, title, fontsize=8.5, fontweight='bold', color=INK, va='top')
    ax.text(x + 0.015, y + h - 0.13, body, fontsize=7.8, color=INK2, va='top', linespacing=1.4)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1, mutation_scale=11))


fig = plt.figure(figsize=(7.2, 3.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
W, H, xs = 0.215, 0.36, [0.015 + i * 0.25 for i in range(4)]
top = [('1  Field recording', 'Ban Nai Thung mudflat\n69 videos, low tide\ntripod camera'),
       ('2  Detection, tracking', 'YOLOv11 detector\nBoT-SORT tracker\n10% box padding'),
       ('3  Bout annotation', '278 sequences\n24 frames, 224 px\nFF 101, MF 80, MW 97'),
       ('4  Grouped 5-fold CV', 'folds split by video\n2 seeds, early stopping\nvideo bootstrap CI')]
bottom = [('8  Reporting', 'balanced accuracy, AUC\nmajority baseline\nexact McNemar tests'),
          ('7  External sessions', 'same pipeline\nblind quality control\nlocked models run once'),
          ('6  Models locked', 'trained on all 278\nSHA-256 hashes\nbefore external data'),
          ('5  Models compared', 'Activity: two-stream,\nR3D-18, EffNet+BiLSTM\nSex: DINOv2 + LR')]
for i, (t, b) in enumerate(top):
    box(ax, xs[i], 0.57, W, H, t, b, accent=(i == 3))
    if i < 3: arrow(ax, xs[i] + W + 0.004, 0.75, xs[i + 1] - 0.004, 0.75)
for i, (t, b) in enumerate(bottom):
    box(ax, xs[i], 0.05, W, H, t, b, accent=(i == 1))
    if i < 3: arrow(ax, xs[i + 1] - 0.004, 0.23, xs[i] + W + 0.004, 0.23)
arrow(ax, xs[3] + W / 2, 0.565, xs[3] + W / 2, 0.415)
save(fig, 'Fig3_workflow')

# ================================================================ Figure 4 - detector comparison
det = pd.read_csv(DATA / 'detector' / 'detector_comparison.csv')
fig, ax = plt.subplots(figsize=(6.4, 2.4))
metrics = [('mAP50_pct', 'mAP@50', '#2a78d6', 'o'), ('precision_pct', 'Precision', '#eb6834', 's'), ('recall_pct', 'Recall', '#1baf7a', 'D')]
labels = [d + (' (selected)' if n == 'selected' else '') for d, n in zip(det.detector, det.note.fillna(''))]
for k, (c, lab, colr, mk) in enumerate(metrics):
    y = np.arange(len(det))[::-1] + (1 - k) * 0.2
    ax.scatter(det[c], y, color=colr, marker=mk, s=42, label=lab, zorder=3, edgecolor='white', linewidth=0.6)
    for xv, yv in zip(det[c], y):
        ax.text(xv + 0.15, yv, f'{xv:.1f}', va='center', fontsize=7.5, color=INK)
ax.set_yticks(np.arange(len(det))[::-1]); ax.set_yticklabels(labels)
ax.set_xlim(90, 100); ax.set_xlabel('Validation performance, %')
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=8)
save(fig, 'Fig4_detector_comparison', det)

# ================================================================ Figure 5 - balanced accuracy with 95% CI
rows = []
for m in ACT_MODELS:
    rows.append(dict(panel='activity', model=m, **summary(cv.y_act, cv[f'{m}|act'], cv.video)))
for m in SEX_MODELS:
    rows.append(dict(panel='sex all', model=m, **summary(cv.y_sex, cv[f'{m}|sex'], cv.video)))
    rows.append(dict(panel='sex feeding', model=m, **summary(cv.y_sex[feed], cv[f'{m}|sex'][feed], cv.video[feed])))
s5 = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.95))
for i, m in enumerate(ACT_MODELS[::-1]):
    r = s5[(s5.panel == 'activity') & (s5.model == m)].iloc[0]
    ax[0].plot([r.ci_low * 100, r.ci_high * 100], [i, i], color=COL[m], lw=2, solid_capstyle='round')
    ax[0].plot(r.balanced_accuracy * 100, i, MARK[m], color=COL[m], ms=7, mec='white', mew=1)
    ax[0].text(101, i, f'{r.balanced_accuracy * 100:.1f}', va='center', fontsize=8, color=INK)
ax[0].set_yticks(range(len(ACT_MODELS))); ax[0].set_yticklabels(ACT_MODELS[::-1]); ax[0].set_title('A  Activity (waving vs feeding)')
for i, m in enumerate(SEX_MODELS[::-1]):
    for off, panel, fill in [(0.17, 'sex all', True), (-0.17, 'sex feeding', False)]:
        r = s5[(s5.panel == panel) & (s5.model == m)].iloc[0]
        ax[1].plot([r.ci_low * 100, r.ci_high * 100], [i + off] * 2, color=COL[m], lw=1.6, solid_capstyle='round')
        ax[1].plot(r.balanced_accuracy * 100, i + off, MARK[m], color=COL[m] if fill else 'white', mec=COL[m], mew=1.4, ms=6.5)
        ax[1].text(101, i + off, f'{r.balanced_accuracy * 100:.1f}', va='center', fontsize=7.5, color=INK if fill else INK2)
ax[1].set_yticks(range(len(SEX_MODELS))); ax[1].set_yticklabels(SEX_MODELS[::-1]); ax[1].set_title('B  Sex (male vs female)')
ax[1].plot([], [], 'o', color=INK2, ms=6, label='all sequences (n = 278)')
ax[1].plot([], [], 'o', color='white', mec=INK2, mew=1.4, ms=6, label='feeding only (n = 181)')
ax[1].legend(loc='lower left', bbox_to_anchor=(-0.02, -0.42), fontsize=7.5, ncol=2, handletextpad=0.3, columnspacing=1)
for a in ax:
    a.axvline(50, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.set_xlim(45, 100); a.set_xlabel('Balanced accuracy, % (95% CI)')
    a.grid(axis='x', color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(axis='y', length=0)
ax[0].text(50.5, len(ACT_MODELS) - 0.55, 'chance', fontsize=7, color=MUTED)
save(fig, 'Fig5_balanced_accuracy', s5[['panel', 'model', 'balanced_accuracy', 'ci_low', 'ci_high']])

# ================================================================ Figure 6 - ROC curves
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4)); rows = []
for a, models, kind, mask, title in [(ax[0], ACT_MODELS, 'act', np.ones(len(cv), bool), 'A  Activity (n = 278)'),
                                     (ax[1], [DINO, EFF, R3D, TSA], 'sex', feed, 'B  Sex, feeding sequences only (n = 181)')]:
    for m in models:
        t, p = cv[f'y_{kind}'].values[mask], cv[f'{m}|{kind}'].values[mask]
        fpr, tpr, _ = roc_curve(t, p)
        a.plot(fpr, tpr, color=COL[m], lw=1.8, ls=LS[m], label=f'{m}  (AUC {roc_auc_score(t, p):.2f})')
        rows += [dict(panel=title, model=m, fpr=f, tpr=s) for f, s in zip(fpr, tpr)]
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 3)))
    a.set_xlabel('False positive rate'); a.set_ylabel('True positive rate'); a.set_title(title)
    a.set_xlim(-0.01, 1.01); a.set_ylim(-0.01, 1.01); a.set_aspect('equal'); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(loc='lower right', fontsize=7, handlelength=2.6)
plt.tight_layout(); save(fig, 'Fig6_ROC', pd.DataFrame(rows))


# ================================================================ Figure 7 - confusion matrices
def cmat(a, t, p, labels, title):
    t, y = np.asarray(t), (np.asarray(p) > THRESHOLD).astype(int)
    M = np.array([[((t == i) & (y == j)).sum() for j in (0, 1)] for i in (0, 1)]); R = M / M.sum(1, keepdims=True)
    a.imshow(R, cmap=BLUES, vmin=0, vmax=1)
    for i in (0, 1):
        for j in (0, 1):
            a.text(j, i, f'{M[i, j]}\n{R[i, j] * 100:.1f}%', ha='center', va='center', fontsize=9, color='white' if R[i, j] > 0.55 else INK)
    a.set_xticks([0, 1]); a.set_xticklabels(labels); a.set_yticks([0, 1]); a.set_yticklabels(labels)
    a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(title); a.tick_params(length=0)
    for s in a.spines.values(): s.set_visible(False)
    return pd.DataFrame(M, index=['true_' + l for l in labels], columns=['pred_' + l for l in labels]).reset_index()


fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9))
d1 = cmat(ax[0], cv.y_act, cv[f'{R3D}|act'], ['Feeding', 'Waving'], 'A  Activity, R3D-18')
d2 = cmat(ax[1], cv.y_sex, cv[f'{DINO}|sex'], ['Female', 'Male'], 'B  Sex, DINOv2 + LR')
plt.tight_layout(); save(fig, 'Fig7_confusion', pd.concat([d1.assign(panel='A'), d2.assign(panel='B')]))

# ================================================================ Figure 8 - external sex test (pre-declared, 71 clips)
GROUPS = [('female', 'feeding', 'Female\nfeeding'), ('male', 'feeding', 'Male\nfeeding'), ('male', 'waving', 'Male\nwaving')]


def strip(a, d, pcol, color, title, ylabel):
    rng = np.random.default_rng(1); rows = []
    for k, (s, act, lab) in enumerate(GROUPS):
        g = d[(d.sex == s) & (d.activity == act)]
        a.scatter(k + rng.uniform(-0.18, 0.18, len(g)), g[pcol], s=22, color=color, edgecolor='white', linewidth=0.6, zorder=3)
        a.text(k, -0.13, f'n = {len(g)}', ha='center', fontsize=7.5, color=INK2)
        rows += [dict(group=lab.replace('\n', ' '), clip=c, p=v) for c, v in zip(g.seq_name, g[pcol])]
    a.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.text(2.45, 0.52, 'threshold', fontsize=7, color=MUTED, ha='right')
    a.set_xticks(range(3)); a.set_xticklabels([g[2] for g in GROUPS]); a.set_ylim(-0.18, 1.05); a.set_xlim(-0.5, 2.5)
    a.set_ylabel(ylabel); a.set_title(title); a.grid(axis='y', color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(axis='x', length=0)
    return pd.DataFrame(rows)


fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1, 1.35]))
d1 = cmat(ax[0], e71.y_sex, e71.p_male, ['Female', 'Male'], f'A  Sex, external E1 (n = {len(e71)})')
d2 = strip(ax[1], e71, 'p_male', COL[DINO], 'B  Predicted probability of male', 'p(male)')
plt.tight_layout(); save(fig, 'Fig8_external_sex', pd.concat([d1.assign(panel='A'), d2.assign(panel='B')]))

# ================================================================ Figure S1 - per-fold balanced accuracy
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True); rows = []
for a, models, kind, title in [(ax[0], ACT_MODELS, 'act', 'A  Activity'), (ax[1], SEX_MODELS, 'sex', 'B  Sex (all sequences)')]:
    for j, m in enumerate(models):
        v = [balanced_accuracy(cv[f'y_{kind}'][cv.fold == k], cv[f'{m}|{kind}'][cv.fold == k]) * 100 for k in range(5)]
        a.plot(np.arange(5) + (j - (len(models) - 1) / 2) * 0.13, v, MARK[m], color=COL[m], ms=6, mec='white', mew=0.8, label=m, ls='none')
        rows += [dict(panel=title, model=m, fold=k + 1, balanced_accuracy=b) for k, b in enumerate(v)]
    a.set_xticks(range(5)); a.set_xticklabels([f'Fold {k + 1}' for k in range(5)]); a.set_title(title)
    a.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.grid(axis='y', color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(axis='x', length=0)
ax[0].set_ylabel('Balanced accuracy, %'); ax[0].set_ylim(40, 101)
h = {l: hh for a in ax for hh, l in zip(*a.get_legend_handles_labels())}
fig.legend(h.values(), h.keys(), loc='lower center', ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'FigS1_per_fold', pd.DataFrame(rows))

# ================================================================ Figure S2 - secondary external activity test
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
d1 = strip(ax[0], e90, 'p_waving_r3d18', COL[R3D], 'A  R3D-18', 'p(waving)')
d2 = strip(ax[1], e90, 'p_waving_aug', COL[TSA], 'B  Two-stream (res. aug.)', '')
plt.tight_layout(); save(fig, 'FigS2_external_activity', pd.concat([d1.assign(panel='A'), d2.assign(panel='B')]))

# ================================================================ Figure S3 - first external run (mismatched extraction)
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9))
d1 = cmat(ax[0], e64.y_sex, e64.p_male, ['Female', 'Male'], f'A  Sex (n = {len(e64)})')
d2 = cmat(ax[1], e64.y_act, e64.p_waving_aug, ['Feeding', 'Waving'], 'B  Activity, two-stream (res. aug.)')
plt.tight_layout(); save(fig, 'FigS3_external_mismatched', pd.concat([d1.assign(panel='A'), d2.assign(panel='B')]))

# ================================================================ Figure S4 - external ROC (90 clips)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4)); rows = []
for m, c in [(R3D, 'p_waving_r3d18'), (TSA, 'p_waving_aug'), (TS, 'p_waving_noaug')]:
    fpr, tpr, _ = roc_curve(e90.y_act, e90[c])
    ax[0].plot(fpr, tpr, color=COL[m], lw=1.8, ls=LS[m], label=f'{m}  (AUC {roc_auc_score(e90.y_act, e90[c]):.2f})')
    rows += [dict(panel='activity', model=m, fpr=f, tpr=s) for f, s in zip(fpr, tpr)]
for lab, msk in [('all clips', np.ones(len(e90), bool)), ('feeding only', (e90.activity == 'feeding').values)]:
    t, p = e90.y_sex.values[msk], e90.p_male.values[msk]; fpr, tpr, _ = roc_curve(t, p)
    ax[1].plot(fpr, tpr, color=COL[DINO], lw=1.8, ls='-' if lab == 'all clips' else (0, (5, 2)), label=f'DINOv2 + LR, {lab}  (AUC {roc_auc_score(t, p):.2f})')
    rows += [dict(panel='sex ' + lab, model=DINO, fpr=f, tpr=s) for f, s in zip(fpr, tpr)]
for a, title in zip(ax, [f'A  Activity (n = {len(e90)}, {int(e90.y_act.sum())} waving)', f'B  Sex (n = {len(e90)})']):
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 3))); a.set_aspect('equal'); a.set_title(title)
    a.set_xlabel('False positive rate'); a.set_ylabel('True positive rate'); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(loc='lower right', fontsize=7, handlelength=2.6)
plt.tight_layout(); save(fig, 'FigS4_external_ROC', pd.DataFrame(rows))
print('\nAll figures in', OUT)
