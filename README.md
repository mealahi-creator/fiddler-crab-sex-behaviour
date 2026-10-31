# Video-based sex and behaviour classification of the fiddler crab *Austruca perplexa*

Code and data to reproduce every table, figure and statistic in:

> Tina, F. W., Mohamed, M. H., Sukkuea, A., Samuel, O. W., & Alahi, M. E. E. *Video-Based Deep Learning for Sex and Behaviour Classification of the Fiddler Crab Austruca perplexa.* Journal of Human, Earth, and Future (under review).

All results in the paper are computed from the **saved per-sequence predictions** of each model in `data/predictions/`. Re-running the analysis takes about 30 seconds on a laptop or in Google Colab, needs no GPU and no video data, and ends with an automatic check against the values printed in the manuscript.

## Quick start

**Google Colab (no installation).** Open `reproduce_in_colab.ipynb` in Colab (*File → Open notebook → GitHub*, paste this repository's URL) and choose *Runtime → Run all*.

**Local computer** (Python 3.9 or newer):

```bash
git clone https://github.com/mealahi-creator/fiddler-crab-sex-behaviour.git
cd fiddler-crab-sex-behaviour
pip install -r requirements.txt
python run_all.py
```

The last step prints `19 of 19 checks passed` when every reproduced value equals the manuscript.

## What is reproduced

| Script | Reproduces | Output |
|---|---|---|
| `scripts/01_results_tables.py` | Tables 3, 5, 6, 7 (McNemar + Holm) and 8; Tables S1, S2, S4, S5; numbers quoted in Sections 4.2–4.4 (fold SD, best-threshold BA, error counts, median probabilities) | `results/tables/*.csv`, `results/RESULTS.md` |
| `scripts/02_tracking_stats.py` | Section 4.1: share of frames detected directly, extrapolated or reused; single-identity sequences | `results/tables/Section4_1_tracking.csv` |
| `scripts/03_figures.py` | Figures 3–8 and S1–S4 (PNG 600 dpi, PDF, editable SVG, and the plotted values as CSV) | `results/figures/` |
| `scripts/04_refit_sex_classifier.py` | Re-trains the DINOv2 / ConvNeXt-T + logistic-regression sex classifiers from the saved frame features under the same video-grouped folds (Table S5, DINOv2 rows of Tables 6–7) | `results/tables/TableS5_refit_from_features.csv` |
| `scripts/05_check_against_manuscript.py` | Compares 19 key values with the manuscript and reports PASS/FAIL | console |

`results/` already contains the outputs, so they can be inspected without running anything.

Figures 1 (study-site map), 2 (example frames) and S5 (original YOLOv11 training curves, unmodified) are images rather than plots of the analysed data; they are provided in `figures_static/`. The detector validation values (Table 1, Figure 4) are taken from the detector training output and stored in `data/detector/detector_comparison.csv`.

## Methods in brief

- **Data:** 278 manually annotated behaviour sequences (101 female feeding, 80 male feeding, 97 male waving) from 69 field videos.
- **Validation:** stratified five-fold cross-validation grouped by source video; the next fold is used for early stopping; two random seeds per model, probabilities averaged.
- **Metrics:** balanced accuracy at a fixed threshold of 0.5; 95% CI from 2,000 bootstrap resamples of videos (percentile method, seed 0); ROC-AUC; majority-class baseline.
- **Model comparison:** exact two-sided McNemar test (binomial test on discordant sequences), Holm-adjusted within each task.
- **External test:** final models locked by SHA-256 hash (`data/plans/LOCK_*.json`) before any external clip was processed; pre-declared sex test on session E1 (71 clips); secondary activity analysis on sessions E1 + E2 (90 clips).

Note on the bootstrap: the cross-validation scripts draw videos from an alphabetically sorted list, while the locked external-test scripts drew them in the order in which they appear in the file. Both are valid random resamples; `src/common.py` keeps both orders (argument `order`) only so that every interval matches the manuscript to the first decimal.

## Repository structure

```
run_all.py                     runs all analysis scripts in order
reproduce_in_colab.ipynb       the same, in Google Colab
src/common.py                  data loading, metrics, bootstrap, McNemar, Holm
scripts/                       01-05 analysis scripts (see table above)
data/predictions/              per-sequence predictions of every model (cross-validation and external tests)
data/tracking/                 complete detector/tracker logs (training sequences and external clips)
data/features/                 frozen DINOv2 and ConvNeXt-T features of 8 frames per sequence
data/detector/                 validation results of the three detectors compared
data/plans/                    pre-declared analysis plans and model lock files (SHA-256 hashes)
results/                       reproduced tables and figures
figures_static/                Figures 1, 2 and S5
original_code/                 Colab code used to process the videos, train the models and run the external tests
```

`data/README.md` describes every data file and column. `original_code/README.md` explains the training and external-test code; running it requires the video sequences and model weights (see below).

## Data not included in this repository

The extracted video sequences (grayscale frames), the external clips, the trained model weights and the raw field videos are too large for GitHub. They are archived at Zenodo: **DOI to be added**. With them, the full pipeline in `original_code/` can be re-run from the videos.

## Licence

Code: MIT licence (see `LICENSE`). Data: Creative Commons Attribution 4.0 (CC BY 4.0). Please cite the article when using either.

## Contact

Md. Eshrat E Alahi, School of Engineering and Technology, Walailak University, Thailand — mdeshrat.al@wu.ac.th
