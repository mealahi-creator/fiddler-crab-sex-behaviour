# Original processing, training and external-test code

These are the Google Colab scripts used to produce the predictions in `data/predictions/`. They are provided so that every step can be inspected. Running them requires the video sequences, external clips and model weights from the Zenodo archive and a GPU runtime. The folder paths at the top of each script point to the authors' Google Drive and must be changed to your own locations.

The analysis in `scripts/` does **not** need this code; it starts from the saved predictions.

## pipeline/ — from video to sequences

| Script | Purpose |
|---|---|
| `Video_Preprocessing_Script.ipynb` | YOLOv11 detection + BoT-SORT tracking of the training videos; gap filling (extrapolation for up to 5 frames, reuse of the last box); 10% box padding; grayscale 224 × 224 crops saved frame by frame; writes the tracking log |
| `external_step0_normalize_frame_rate.py` | Converts variable-frame-rate external videos to a constant 30 fps; flags videos with a real frame rate below 25 fps |
| `external_step1_track_preview.py` | Runs the same detector and tracker on each external video and makes a preview with numbered crabs, used for bout annotation |
| `external_qc_make_contact_sheets.py` | Builds shuffled contact sheets (no labels, no predictions) for the blind quality-control scorer |
| `external_qc_lock.py` | Locks the scorer's exclusions before any prediction is made |

## training/ — video-grouped five-fold cross-validation and final models

| Script | Purpose |
|---|---|
| `twostream_grouped_cv.py` | Two-stream 3D CNN (with and without resolution augmentation), 5 folds × 2 seeds |
| `baselines_r3d18_effnet_grouped_cv.py` | R3D-18 (Kinetics-400) and EfficientNet-B0 + BiLSTM (ImageNet) under the same folds, seeds and augmentation |
| `sex_probe_colab.py` | DINOv2 / ConvNeXt-T + logistic-regression sex classifiers (8 pre-declared variants) |
| `sex_probe_save_features.py` | Saves the frozen frame features used by `scripts/04_refit_sex_classifier.py` |
| `final_train_lock_twostream_dinov2.py` | Trains the final two-stream and DINOv2 models on all 278 sequences and records their SHA-256 hashes |
| `final_train_lock_r3d18.py` | Trains and locks the final R3D-18 activity model |

## external/ — tests with locked models (each run once)

| Script | Test |
|---|---|
| `external_test_first_run_64clips.py` | First run on session E1 with clips extracted differently from the training data (Table S2) |
| `external_test_primary_71clips.py` | Pre-declared test on session E1 (Table 8) |
| `external_test_secondary_90clips_r3d18.py` | Locked R3D-18 on all external waving clips from E1 and E2 (Tables 8 and S1) |

Every external script checks the model files against the SHA-256 hashes in `data/plans/` before predicting.
